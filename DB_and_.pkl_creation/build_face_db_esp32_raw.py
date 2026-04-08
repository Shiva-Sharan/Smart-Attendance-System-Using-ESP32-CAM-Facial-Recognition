import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


# ==================================================
# DEFAULT PATHS
# ==================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, "Faces")
DEFAULT_DET_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.onnx")
DEFAULT_RECOG_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "w600k_mbf.onnx")
DEFAULT_OUTPUT_PKL = os.path.join(PROJECT_ROOT, "face_db.pkl")


# ==================================================
# DEFAULT TUNING
# ==================================================
DEFAULT_CROP_PADDING_RATIO = 1.25
DEFAULT_MIN_DET_CONF = 0.70
DEFAULT_MIN_FACE_SIZE = 90
DEFAULT_MIN_BLUR_SCORE = 65.0

# Required ESP32 profile thresholds
DEFAULT_STRICT_SELF_MATCH = 0.55
DEFAULT_MAX_OTHER_MATCH = 0.75

# Alignment parameters
ALIGN_EXPAND_RATIO = 1.10
MAX_ALIGN_ROT_DEG = 18.0
MIN_ALIGN_ROT_DEG = 2.0
MIN_MASK_PIXELS = 120


QNN_PROVIDER_OPTIONS = {
    "backend_path": "QnnHtp.dll",
    "htp_performance_mode": "high_performance",
    "rpc_control_latency": "low",
}


@dataclass
class SampleEmbedding:
    filename: str
    embedding: np.ndarray
    det_conf: float
    blur: float
    crop_size: int
    align_rot_deg: float


# ==================================================
# MODEL + BASIC HELPERS
# ==================================================
def make_session(model_path: str, label: str) -> ort.InferenceSession:
    providers = []
    available = ort.get_available_providers()
    if "QNNExecutionProvider" in available:
        providers.append(("QNNExecutionProvider", QNN_PROVIDER_OPTIONS))
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, providers=providers)
    print(f"[{label}] providers: {session.get_providers()}")
    print(f"[{label}] input type: {session.get_inputs()[0].type}")
    return session


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        return vec / norm
    return vec


def blur_score(face_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def iter_people(base_path: str) -> Iterable[Tuple[str, str]]:
    for entry in sorted(os.scandir(base_path), key=lambda e: e.name):
        if entry.is_dir():
            yield entry.name, entry.path


def iter_images(person_dir: str) -> Iterable[Tuple[str, str]]:
    allowed = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    for entry in sorted(os.scandir(person_dir), key=lambda e: e.name):
        if entry.is_file() and entry.name.lower().endswith(allowed):
            yield entry.name, entry.path


# ==================================================
# ESP32 RUNTIME-COMPATIBLE RECOG PREPROCESS
# ==================================================
class RuntimeCompatibleEmbedder:
    """
    Mirrors runtime preprocessing flow:
      - resize to 112x112 (INTER_AREA)
      - BGR -> RGB
      - x * (1/127.5) - 1.0
      - HWC -> CHW
    """

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.input_type = session.get_inputs()[0].type
        self.use_f16 = "float16" in self.input_type

        self.inv_127_5 = np.float32(1.0 / 127.5)
        self.resize_bgr = np.empty((112, 112, 3), dtype=np.uint8)
        self.resize_rgb = np.empty((112, 112, 3), dtype=np.uint8)
        self.norm_hwc = np.empty((112, 112, 3), dtype=np.float32)
        self.input_f32 = np.empty((1, 3, 112, 112), dtype=np.float32)
        self.input_f16 = np.empty((1, 3, 112, 112), dtype=np.float16) if self.use_f16 else None

    def _prepare_input(self, face_bgr: np.ndarray) -> np.ndarray:
        cv2.resize(face_bgr, (112, 112), dst=self.resize_bgr, interpolation=cv2.INTER_AREA)
        cv2.cvtColor(self.resize_bgr, cv2.COLOR_BGR2RGB, dst=self.resize_rgb)

        np.multiply(self.resize_rgb, self.inv_127_5, out=self.norm_hwc, casting="unsafe")
        np.subtract(self.norm_hwc, 1.0, out=self.norm_hwc)

        self.input_f32[0] = self.norm_hwc.transpose(2, 0, 1)
        if self.input_f16 is not None:
            np.copyto(self.input_f16, self.input_f32, casting="unsafe")
            return self.input_f16
        return self.input_f32

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        model_input = self._prepare_input(face_bgr)
        raw = self.session.run(None, {self.input_name: model_input})[0][0]
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        return normalize(raw)


# ==================================================
# DETECTOR HELPERS
# ==================================================
def decode_face(hm: np.ndarray, bx: np.ndarray, frame_w: int, frame_h: int, avg_brightness: float):
    hm = hm[0, 0].astype(np.float32)
    bx = bx[0].astype(np.float32)

    threshold = 160 if avg_brightness > 90 else 145
    _, max_val, _, max_loc = cv2.minMaxLoc(hm)
    if max_val <= threshold:
        return None

    x, y = max_loc
    reg = bx[:, y, x]

    cx = (x + 0.5) * (frame_w / 80)
    cy = (y + 0.5) * (frame_h / 60)
    bw = (reg[2] / 255) * frame_w * 2.2
    bh = (reg[3] / 255) * frame_h * 2.8

    bbox = [int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)]
    conf = float(max_val / 255.0)
    return bbox, conf


def crop_square_with_padding(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    crop_padding_ratio: float,
) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox

    side = int(max(bw, bh) * crop_padding_ratio)
    center_x = x + (bw / 2.0)
    center_y = y + (bh / 2.0)

    new_x = max(0, int(center_x - (side / 2.0)))
    new_y = max(0, int(center_y - (side / 2.0)))

    if new_x + side > w:
        side = w - new_x
    if new_y + side > h:
        side = h - new_y

    if side <= 0:
        return None

    crop = frame[new_y:new_y + side, new_x:new_x + side]
    if crop.size == 0:
        return None
    return crop


def detect_face_crop(
    frame: np.ndarray,
    det_session: ort.InferenceSession,
    det_input_name: str,
    det_input_tensor: np.ndarray,
    min_det_conf: float,
    crop_padding_ratio: float,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[Tuple[int, int, int, int]], str]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(cv2.mean(gray)[0])

    cv2.resize(gray, (640, 480), dst=det_input_tensor[0, 0], interpolation=cv2.INTER_LINEAR)
    out = det_session.run(None, {det_input_name: det_input_tensor})
    decoded = decode_face(out[0], out[1], frame.shape[1], frame.shape[0], avg_brightness)
    if not decoded:
        return None, None, None, "no face"

    raw_bbox, conf = decoded
    if conf < min_det_conf:
        return None, conf, tuple(raw_bbox), f"low det conf ({conf:.3f} < {min_det_conf})"

    crop = crop_square_with_padding(frame, tuple(raw_bbox), crop_padding_ratio)
    if crop is None:
        return None, conf, tuple(raw_bbox), "invalid crop"

    return crop, conf, tuple(raw_bbox), "ok"


# ==================================================
# ALIGNMENT HELPERS
# ==================================================
def _largest_component(mask: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float], int]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    cx, cy = centroids[idx]

    component_mask = np.zeros_like(mask, dtype=np.uint8)
    component_mask[labels == idx] = 255
    return component_mask, (x, y, w, h), (float(cx), float(cy)), area


def _build_best_face_mask(gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float]]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)

    h, w = gray.shape[:2]
    diag = float(np.hypot(max(1.0, w / 2.0), max(1.0, h / 2.0)))
    candidates = []

    for cand in (otsu, cv2.bitwise_not(otsu)):
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel, iterations=1)
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel, iterations=2)
        comp = _largest_component(cand)
        if comp is None:
            continue

        comp_mask, bbox, center, area = comp
        area_ratio = area / float(max(1, h * w))
        cx, cy = center
        center_penalty = float(np.hypot(cx - (w / 2.0), cy - (h / 2.0)) / max(diag, 1e-6))
        score = abs(area_ratio - 0.45) + (0.65 * center_penalty)

        if area_ratio < 0.03 or area_ratio > 0.98:
            score += 1.0

        candidates.append((score, comp_mask, bbox, center))

    if not candidates:
        full_mask = np.full_like(gray, 255, dtype=np.uint8)
        return full_mask, (0, 0, w, h), (w / 2.0, h / 2.0)

    candidates.sort(key=lambda item: item[0])
    _, best_mask, best_bbox, best_center = candidates[0]
    return best_mask, best_bbox, best_center


def _estimate_rotation_deg(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size < MIN_MASK_PIXELS:
        return 0.0

    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    pts -= np.mean(pts, axis=0, keepdims=True)

    cov = (pts.T @ pts) / float(max(1, pts.shape[0] - 1))
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    axis = eig_vecs[:, int(np.argmax(eig_vals))]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))

    to_pos_vertical = 90.0 - angle
    to_neg_vertical = -90.0 - angle
    rotate = to_pos_vertical if abs(to_pos_vertical) < abs(to_neg_vertical) else to_neg_vertical
    rotate = float(np.clip(rotate, -MAX_ALIGN_ROT_DEG, MAX_ALIGN_ROT_DEG))

    if abs(rotate) < MIN_ALIGN_ROT_DEG:
        return 0.0
    return rotate


def _safe_square_crop(image: np.ndarray, center_xy: Tuple[float, float], side_len: int) -> np.ndarray:
    h, w = image.shape[:2]
    side = int(np.clip(side_len, 1, min(h, w)))
    cx, cy = center_xy

    x0 = int(round(cx - (side / 2.0)))
    y0 = int(round(cy - (side / 2.0)))
    x0 = max(0, min(x0, w - side))
    y0 = max(0, min(y0, h - side))

    return image[y0:y0 + side, x0:x0 + side]


def align_face_crop(face_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    mask, _, center = _build_best_face_mask(gray)
    rotate_deg = _estimate_rotation_deg(mask)

    aligned = face_bgr
    if rotate_deg != 0.0:
        h, w = face_bgr.shape[:2]
        rot_mat = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
        aligned = cv2.warpAffine(
            face_bgr,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    _, bbox, center = _build_best_face_mask(aligned_gray)
    _, _, bw, bh = bbox

    side = int(round(max(bw, bh) * ALIGN_EXPAND_RATIO))
    side = max(side, 1)

    crop = _safe_square_crop(aligned, center, side)
    if crop.size == 0:
        h, w = aligned.shape[:2]
        fallback_side = min(h, w)
        crop = _safe_square_crop(aligned, (w / 2.0, h / 2.0), fallback_side)

    return crop, rotate_deg


def check_quality(face_bgr: np.ndarray, min_face_size: int, min_blur_score: float) -> Tuple[bool, str, float, int]:
    h, w = face_bgr.shape[:2]
    min_side = int(min(h, w))
    if min_side < min_face_size:
        return False, f"face too small ({min_side}px < {min_face_size}px)", 0.0, min_side

    blur = blur_score(face_bgr)
    if blur < min_blur_score:
        return False, f"too blurry ({blur:.1f} < {min_blur_score})", blur, min_side

    return True, "ok", blur, min_side


# ==================================================
# STEP 1: DETECT + CROP + ALIGN + EMBED
# ==================================================
def build_candidate_embeddings(
    dataset_path: str,
    det_session: ort.InferenceSession,
    det_input_name: str,
    embedder: RuntimeCompatibleEmbedder,
    min_det_conf: float,
    crop_padding_ratio: float,
    min_face_size: int,
    min_blur_score: float,
) -> Dict[str, List[SampleEmbedding]]:
    person_data: Dict[str, List[SampleEmbedding]] = {}
    det_input = np.empty((1, 1, 480, 640), dtype=np.uint8)

    print("\n" + "=" * 72)
    print("STEP 1: Detect, square-pad crop, align, quality filter, embed")
    print("=" * 72)
    print("Dataset mode: RAW ESP32 IMAGES ONLY (no edited/background-removed input)")

    for person_id, person_dir in iter_people(dataset_path):
        samples: List[SampleEmbedding] = []
        rejected = 0
        seen = 0

        print(f"\n[{person_id}] scanning...")

        for img_name, img_path in iter_images(person_dir):
            seen += 1
            frame = cv2.imread(img_path)
            if frame is None:
                rejected += 1
                print(f"  Rejected {img_name}: unreadable image")
                continue

            crop, det_conf, raw_bbox, det_reason = detect_face_crop(
                frame=frame,
                det_session=det_session,
                det_input_name=det_input_name,
                det_input_tensor=det_input,
                min_det_conf=min_det_conf,
                crop_padding_ratio=crop_padding_ratio,
            )

            if crop is None:
                rejected += 1
                if det_conf is None:
                    print(f"  Rejected {img_name}: {det_reason}")
                else:
                    print(f"  Rejected {img_name}: {det_reason}, bbox={raw_bbox}")
                continue

            aligned_face, rotate_deg = align_face_crop(crop)

            ok, q_reason, blur, crop_size = check_quality(
                aligned_face,
                min_face_size=min_face_size,
                min_blur_score=min_blur_score,
            )
            if not ok:
                rejected += 1
                print(f"  Rejected {img_name}: {q_reason}")
                continue

            emb = embedder.embed(aligned_face)
            samples.append(
                SampleEmbedding(
                    filename=img_name,
                    embedding=emb,
                    det_conf=float(det_conf) if det_conf is not None else 0.0,
                    blur=blur,
                    crop_size=crop_size,
                    align_rot_deg=rotate_deg,
                )
            )

            print(
                f"  Kept {img_name}: det={det_conf:.3f}, blur={blur:.1f}, "
                f"crop={crop_size}px, align_rot={rotate_deg:.1f}deg"
            )

        person_data[person_id] = samples
        print(f"[{person_id}] kept={len(samples)} rejected={rejected} total={seen}")

    return person_data


def robust_median_embedding(emb_matrix: np.ndarray) -> np.ndarray:
    median_emb = np.median(emb_matrix, axis=0)
    return normalize(median_emb.astype(np.float32, copy=False))


def compute_rough_profile_medians(person_data: Dict[str, List[SampleEmbedding]]) -> Dict[str, np.ndarray]:
    rough: Dict[str, np.ndarray] = {}
    for person_id, samples in person_data.items():
        if not samples:
            continue
        matrix = np.stack([s.embedding for s in samples], axis=0)
        rough[person_id] = robust_median_embedding(matrix)
    return rough


# ==================================================
# STEP 2: STRICT FILTERING
# ==================================================
def strict_filter(
    person_data: Dict[str, List[SampleEmbedding]],
    rough_profiles: Dict[str, np.ndarray],
    strict_self_match: float,
    max_other_match: float,
) -> Dict[str, np.ndarray]:
    final_db: Dict[str, np.ndarray] = {}

    print("\n" + "=" * 72)
    print("STEP 2: Strict embedding filtering")
    print("=" * 72)
    print(
        f"Thresholds: self_match>={strict_self_match}, other_match<={max_other_match}"
    )

    for person_id, samples in person_data.items():
        if not samples:
            print(f"\n[{person_id}] removed: no candidate embeddings")
            continue

        own_profile = rough_profiles[person_id]
        emb_matrix = np.stack([s.embedding for s in samples], axis=0)
        filenames = [s.filename for s in samples]

        other_ids = [pid for pid in rough_profiles.keys() if pid != person_id]
        other_matrix = (
            np.stack([rough_profiles[pid] for pid in other_ids], axis=0)
            if other_ids
            else None
        )

        self_sims = emb_matrix @ own_profile
        if other_matrix is not None:
            other_sims = emb_matrix @ other_matrix.T
            max_other = np.max(other_sims, axis=1)
            max_other_idx = np.argmax(other_sims, axis=1)
        else:
            max_other = np.full((emb_matrix.shape[0],), -1.0, dtype=np.float32)
            max_other_idx = np.full((emb_matrix.shape[0],), -1, dtype=np.int32)

        keep_mask = self_sims >= strict_self_match
        if other_matrix is not None:
            keep_mask &= max_other <= max_other_match

        kept_embeddings: List[np.ndarray] = []
        print(f"\n[{person_id}] evaluating {len(samples)} embeddings")

        for i, keep in enumerate(keep_mask):
            if keep:
                kept_embeddings.append(emb_matrix[i])
                continue

            reasons: List[str] = []
            if self_sims[i] < strict_self_match:
                reasons.append(
                    f"low self-match ({self_sims[i]:.3f} < {strict_self_match})"
                )

            if other_matrix is not None and max_other[i] > max_other_match:
                conflict_id = other_ids[int(max_other_idx[i])]
                reasons.append(
                    f"similar to '{conflict_id}' ({max_other[i]:.3f} > {max_other_match})"
                )

            reason_text = " | ".join(reasons) if reasons else "strict filter"
            print(f"  Rejected {filenames[i]}: {reason_text}")

        if not kept_embeddings:
            print(f"  Removed {person_id}: all embeddings rejected")
            continue

        kept_matrix = np.stack(kept_embeddings, axis=0)
        final_profile = robust_median_embedding(kept_matrix)
        final_db[person_id] = final_profile
        print(f"  Kept {len(kept_embeddings)}/{len(samples)} embeddings")

    return final_db


def save_database(face_db: Dict[str, np.ndarray], output_path: str) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(face_db, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build face embedding database from raw ESP32 images using ONNX detection+recognition."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Dataset root path")
    parser.add_argument("--det-model", default=DEFAULT_DET_MODEL_PATH, help="Face detection ONNX path")
    parser.add_argument("--recog-model", default=DEFAULT_RECOG_MODEL_PATH, help="Recognition ONNX path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PKL, help="Output .pkl path")

    # Future tuning parameters
    parser.add_argument("--min-det-conf", type=float, default=DEFAULT_MIN_DET_CONF, help="Detection confidence threshold")
    parser.add_argument("--strict-self-match", type=float, default=DEFAULT_STRICT_SELF_MATCH, help="Self-similarity threshold")
    parser.add_argument("--max-other-match", type=float, default=DEFAULT_MAX_OTHER_MATCH, help="Cross-person similarity threshold")
    parser.add_argument("--crop-padding-ratio", type=float, default=DEFAULT_CROP_PADDING_RATIO, help="Square crop padding ratio")
    parser.add_argument("--min-face-size", type=int, default=DEFAULT_MIN_FACE_SIZE, help="Minimum aligned face size in pixels")
    parser.add_argument("--min-blur-score", type=float, default=DEFAULT_MIN_BLUR_SCORE, help="Minimum Laplacian blur score")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isdir(args.dataset):
        print(f"ERROR: dataset folder not found: {args.dataset}")
        return 1

    if not os.path.isfile(args.det_model):
        print(f"ERROR: detection model not found: {args.det_model}")
        return 1

    if not os.path.isfile(args.recog_model):
        print(f"ERROR: recognition model not found: {args.recog_model}")
        return 1

    print("Loading ONNX models...")
    det_session = make_session(args.det_model, "DETECT")
    recog_session = make_session(args.recog_model, "RECOG")
    det_input_name = det_session.get_inputs()[0].name
    embedder = RuntimeCompatibleEmbedder(recog_session)

    person_data = build_candidate_embeddings(
        dataset_path=args.dataset,
        det_session=det_session,
        det_input_name=det_input_name,
        embedder=embedder,
        min_det_conf=args.min_det_conf,
        crop_padding_ratio=args.crop_padding_ratio,
        min_face_size=args.min_face_size,
        min_blur_score=args.min_blur_score,
    )

    rough_profiles = compute_rough_profile_medians(person_data)
    final_db = strict_filter(
        person_data=person_data,
        rough_profiles=rough_profiles,
        strict_self_match=args.strict_self_match,
        max_other_match=args.max_other_match,
    )

    print("\n" + "=" * 72)
    print("STEP 3: Save final database")
    print("=" * 72)
    save_database(final_db, args.output)

    input_people = len(person_data)
    with_candidates = sum(1 for samples in person_data.values() if samples)
    print(f"Input persons: {input_people}")
    print(f"Persons with candidate embeddings: {with_candidates}")
    print(f"Final DB profiles: {len(final_db)}")
    print(f"Saved database: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
