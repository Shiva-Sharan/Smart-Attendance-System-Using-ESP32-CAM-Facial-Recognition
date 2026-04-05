import os
import cv2
import numpy as np
import onnxruntime as ort
import pickle
import shutil

# ==================================================
# PATHS
# ==================================================
FACE_MODEL_PATH = r"D:\Major Project\models\model.onnx"
RECOG_MODEL_PATH = r"D:\Major Project\models\w600k_mbf.onnx"
FACE_DB_PKL = r"D:\Major Project\face_db.pkl"

DATASET_PATH = r"D:\Major Project\Faces"
CROPPED_PATH = r"D:\Major Project\cropped_faces"

# ==================================================
# OPTIMIZED TUNING PARAMETERS
# ==================================================
CROP_PADDING_RATIO = 1.25
MIN_DETECTION_CONF = 0.75
MIN_FACE_SIZE = 90
MIN_BLUR_SCORE = 65.0

STRICT_SELF_MATCH = 0.75
MAX_OTHER_MATCH = 0.60

qnn = {"backend_path": "QnnHtp.dll"}


# ==================================================
# LOAD MODELS
# ==================================================
print("Loading models into memory...")


def make_session(model_path):
    providers = []
    available = ort.get_available_providers()
    if "QNNExecutionProvider" in available:
        providers.append(("QNNExecutionProvider", qnn))
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(model_path, providers=providers)


face_sess = make_session(FACE_MODEL_PATH)
face_input = face_sess.get_inputs()[0].name

recog_sess = make_session(RECOG_MODEL_PATH)
recog_input = recog_sess.get_inputs()[0].name

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ==================================================
# HELPERS
# ==================================================
def blur_score(face):
    return cv2.Laplacian(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def preprocess_recog(face):
    face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = (face.astype(np.float32) / 255.0 - 0.5) / 0.5
    return face.transpose(2, 0, 1)[None]


def decode_face(hm, bx, fw, fh, avg):
    hm = hm[0, 0].astype(np.float32)
    bx = bx[0].astype(np.float32)

    threshold = 160 if avg > 90 else 145
    _, max_val, _, max_loc = cv2.minMaxLoc(hm)
    if max_val <= threshold:
        return None

    x, y = max_loc
    reg = bx[:, y, x]

    cx = (x + 0.5) * (fw / 80)
    cy = (y + 0.5) * (fh / 60)
    bw = (reg[2] / 255) * fw * 2.2
    bh = (reg[3] / 255) * fh * 2.8

    return [int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)], float(max_val / 255.0)


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def iter_people(base_path):
    for entry in sorted(os.scandir(base_path), key=lambda e: e.name):
        if entry.is_dir():
            yield entry.name, entry.path


def iter_images(person_dir):
    allowed = (".png", ".jpg", ".jpeg")
    for entry in sorted(os.scandir(person_dir), key=lambda e: e.name):
        if entry.is_file() and entry.name.lower().endswith(allowed):
            yield entry.name, entry.path


# ==================================================
# MAIN LOGIC
# ==================================================
def main():
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Cannot find folder: {DATASET_PATH}")
        return

    if os.path.exists(CROPPED_PATH):
        shutil.rmtree(CROPPED_PATH)
    os.makedirs(CROPPED_PATH, exist_ok=True)

    person_data = {}
    det_input = np.empty((1, 1, 480, 640), dtype=np.uint8)

    print("\n" + "=" * 50)
    print("STEP 1: Padded square extraction and pre-filtering")
    print("=" * 50)

    for person_id, person_dir in iter_people(DATASET_PATH):
        os.makedirs(os.path.join(CROPPED_PATH, person_id), exist_ok=True)
        person_data[person_id] = []

        for img_name, img_path in iter_images(person_dir):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = float(cv2.mean(gray)[0])
            gray_proc = clahe.apply(gray) if avg_brightness < 70 else gray

            cv2.resize(gray_proc, (640, 480), dst=det_input[0, 0], interpolation=cv2.INTER_LINEAR)
            out = face_sess.run(None, {face_input: det_input})
            res = decode_face(out[0], out[1], w, h, avg_brightness)

            if not res:
                continue

            raw, conf = res
            if conf <= MIN_DETECTION_CONF:
                continue

            x, y, bw, bh = raw
            side = int(max(bw, bh) * CROP_PADDING_RATIO)
            center_x = x + (bw / 2.0)
            center_y = y + (bh / 2.0)

            new_x = max(0, int(center_x - (side / 2.0)))
            new_y = max(0, int(center_y - (side / 2.0)))
            if new_x + side > w:
                side = w - new_x
            if new_y + side > h:
                side = h - new_y

            if side <= 0:
                continue

            face_crop = frame[new_y:new_y + side, new_x:new_x + side]
            if face_crop.size == 0:
                continue

            if side < MIN_FACE_SIZE:
                print(f"  Skipped {img_name}: Face too small ({side}px)")
                continue

            if blur_score(face_crop) < MIN_BLUR_SCORE:
                print(f"  Skipped {img_name}: Photo is too blurry")
                continue

            crop_save_path = os.path.join(CROPPED_PATH, person_id, img_name)
            cv2.imwrite(crop_save_path, face_crop)

            raw_emb = recog_sess.run(None, {recog_input: preprocess_recog(face_crop)})[0][0]
            person_data[person_id].append((img_name, normalize(raw_emb)))

        print(f"ID {person_id[-4:]}: Extracted {len(person_data[person_id])} high-quality padded faces")

    print("\n" + "=" * 50)
    print("STEP 2: Strict quality and anti-confusion filtering")
    print("=" * 50)

    rough_means = {}
    for pid, data_list in person_data.items():
        if data_list:
            embs = [item[1] for item in data_list]
            rough_means[pid] = normalize(np.mean(embs, axis=0))

    final_db = {}

    for person_id, data_list in person_data.items():
        if not data_list:
            continue

        my_rough_mean = rough_means[person_id]
        good_embs = []
        print(f"\nEvaluating ID: {person_id[-4:]}")

        for filename, emb in data_list:
            self_sim = float(np.dot(my_rough_mean, emb))
            if self_sim < STRICT_SELF_MATCH:
                print(f"  Rejected '{filename}' -> Low self-match ({self_sim:.3f})")
                continue

            looks_like_stranger = False
            for other_id, other_mean in rough_means.items():
                if other_id == person_id:
                    continue
                stranger_sim = float(np.dot(other_mean, emb))
                if stranger_sim > MAX_OTHER_MATCH:
                    print(
                        f"  Conflict '{filename}' -> Similar to ID {other_id[-4:]} ({stranger_sim:.3f})"
                    )
                    looks_like_stranger = True
                    break

            if not looks_like_stranger:
                good_embs.append(emb)

        if good_embs:
            final_mean = normalize(np.mean(good_embs, axis=0))
            final_db[person_id] = final_mean
            print(f"  Success: Kept {len(good_embs)} validated photos")
        else:
            print(f"  Failed: All photos rejected, ID {person_id[-4:]} removed")

    print("\n" + "=" * 50)
    print("STEP 3: Writing final database")
    print("=" * 50)

    with open(FACE_DB_PKL, "wb") as f:
        pickle.dump(final_db, f)

    print(f"Complete. Saved {len(final_db)} distinct profiles to {FACE_DB_PKL}")
    print(f"Review extracted crops in: {CROPPED_PATH}")


if __name__ == "__main__":
    main()
