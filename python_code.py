import cv2
import numpy as np
import onnxruntime as ort
import time
import pickle
import os
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread, Lock, Event
from collections import deque
import sqlite3
from datetime import datetime
import paho.mqtt.client as mqtt
import urllib.request
from urllib.parse import urlsplit
from queue import Queue, Full, Empty

# ==================================================
# PATHS & CONSTANTS
# ==================================================
FACE_MODEL_PATH = r"D:\Major Project\models\model.onnx"
LIVENESS_MODEL_PATH = r"D:\Major Project\models\best_model_npu_final.onnx"
RECOG_MODEL_PATH = r"D:\Major Project\models\w600k_mbf.onnx"
FACE_DB_PKL = r"D:\Major Project\face_db.pkl"
FACES_DATASET_PATH = r"D:\Major Project\Faces"
DB_NAME = r"D:\Major Project\attendance.db"
DB_FALLBACK_NAME = r"D:\Major Project\attendance_runtime.db"
DB_MEMORY_URI = "file:edgeid_runtime?mode=memory&cache=shared"
ATTENDANCE_DATE_FORMAT = "%d %b %Y"
ATTENDANCE_DEFAULT_SESSION = "Daily"
ATTENDANCE_DEFAULT_TIME = "00:00:00"
ATTENDANCE_STATUS_ABSENT = "Absent"
ATTENDANCE_STATUS_PRESENT = "Present"

# --- HARDWARE IPs AND TOPICS ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_LCD_TOPIC = "shiva_edgeid/lcd_display"
ESP32_CAM_URL = "http://172.27.99.90:81/stream" # <-- REPLACE WITH YOUR ESP32 IP ADDRESS
ESP32_STREAM_TIMEOUT = 8.0
ESP32_RECONNECT_DELAY = 1.0
ESP32_CHUNK_SIZE = 4096
ESP32_MAX_BUFFER_BYTES = 2 * 1024 * 1024
FACE_DET_EVERY_N = 2
GUI_REFRESH_MS = 16
DISPLAY_MAX_WIDTH = 960
MQTT_RECONNECT_MIN_SEC = 1
MQTT_RECONNECT_MAX_SEC = 8
MQTT_PUBLISH_QOS = 1
DB_QUEUE_MAX = 256

# --- THRESHOLDS ---
# Fallback values; runtime values are calibrated from FACE_DB_PKL when possible.
SIM_ACCEPT_TH = 0.40        # fallback confident accept cutoff
SIM_UNCERTAIN_TH = 0.62    # fallback uncertain cutoff
SIM_MARGIN_TH = 0.035       # best must beat second-best by this margin
AUTO_CALIBRATE_SIM = True
ID_CONFIRM_FRAMES = 2       # require same confident identity this many times

CROP_PADDING_RATIO = 1.25
LIVENESS_THRESHOLD = 0.65

MIN_FACE_RATIO = 0.12
MAX_FACE_RATIO = 0.45

VOTE_WINDOW = 12
VOTE_MIN_LIVE = 10
MIN_CONSECUTIVE_LIVE = 8

STRICT_WINDOW = 10

BASE_BLUR = 60.0
LOW_LIGHT_WARN_TH = 55
EXTREME_DARK_TH = 30  # only trigger flash in near pitch-black conditions
CLAHE_TH = 70

qnn = {"backend_path": "QnnHtp.dll"}
COOLDOWN_SET = set()
cooldown_lock = Lock()
stop_event = Event()
active_db_path = DB_NAME
memory_anchor_conn = None
daily_init_lock = Lock()
daily_init_cache_key = None
cooldown_day_key = None

cv2.setUseOptimized(True)

# ==================================================
# HARDWARE CONTROL FUNCTIONS (MQTT)
# ==================================================
current_lcd_msg = ""
lcd_lock = Lock()
lcd_queue = Queue(maxsize=8)
mqtt_connected = False


def _on_mqtt_connect(client, userdata, flags, reason_code, properties=None):
    del client, userdata, flags, properties
    global mqtt_connected
    reason_value = getattr(reason_code, "value", reason_code)
    try:
        reason_value = int(reason_value)
    except (TypeError, ValueError):
        reason_value = None

    mqtt_connected = reason_value == 0
    if mqtt_connected:
        print(f"Connected to MQTT Broker: {MQTT_BROKER}")
    else:
        print(f"MQTT connect failed with code: {reason_code}")


def _on_mqtt_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    del client, userdata, disconnect_flags, properties
    global mqtt_connected
    mqtt_connected = False
    if not stop_event.is_set():
        print(f"MQTT disconnected. reason={reason_code}")


def _lcd_publish_worker():
    pending_message = None

    while not stop_event.is_set():
        if pending_message is None:
            try:
                pending_message = lcd_queue.get(timeout=0.2)
            except Empty:
                continue

        # Keep only the latest message while waiting; this avoids stale LCD updates.
        while True:
            try:
                pending_message = lcd_queue.get_nowait()
            except Empty:
                break

        if not mqtt_client or not mqtt_connected:
            time.sleep(0.05)
            continue

        try:
            result = mqtt_client.publish(MQTT_LCD_TOPIC, pending_message, qos=MQTT_PUBLISH_QOS)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"LCD Update:\n{pending_message}")
                pending_message = None
            else:
                time.sleep(0.05)
        except Exception:
            time.sleep(0.05)


try:
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = _on_mqtt_connect
    mqtt_client.on_disconnect = _on_mqtt_disconnect
    mqtt_client.reconnect_delay_set(MQTT_RECONNECT_MIN_SEC, MQTT_RECONNECT_MAX_SEC)
    mqtt_client.connect_async(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    print(f"MQTT client initialization failed: {e}")
    mqtt_client = None


def update_lcd(message):
    global current_lcd_msg
    if not message:
        return

    with lcd_lock:
        if message == current_lcd_msg:
            return
        current_lcd_msg = message

    try:
        lcd_queue.put_nowait(message)
    except Full:
        try:
            lcd_queue.get_nowait()
        except Empty:
            pass
        try:
            lcd_queue.put_nowait(message)
        except Full:
            pass


Thread(target=_lcd_publish_worker, daemon=True).start()

ESP32_IP = urlsplit(ESP32_CAM_URL).hostname or ""
ESP32_FLASH_BASE_URL = f"http://{ESP32_IP}" if ESP32_IP else ""
flash_state_lock = Lock()
flash_is_on = None
flash_pending_state = None
flash_request_in_flight = False


def _send_flash_request(turn_on):
    global flash_request_in_flight, flash_is_on, flash_pending_state
    if not ESP32_FLASH_BASE_URL:
        with flash_state_lock:
            flash_request_in_flight = False
        return

    val = 1 if turn_on else 0
    url = f"{ESP32_FLASH_BASE_URL}/control?var=led_intensity&val={val}"
    try:
        urllib.request.urlopen(url, timeout=0.25).read(1)
        print("FLASH ON (dark)" if turn_on else "FLASH OFF")
    except Exception:
        pass
    finally:
        next_state = None
        with flash_state_lock:
            flash_request_in_flight = False
            # If brightness flipped during in-flight request, send latest pending state next.
            if flash_pending_state is not None and flash_pending_state != flash_is_on:
                next_state = flash_pending_state
                flash_pending_state = None
                flash_is_on = next_state
                flash_request_in_flight = True
            else:
                flash_pending_state = None

        if next_state is not None:
            Thread(target=_send_flash_request, args=(next_state,), daemon=True).start()

def update_flash_by_brightness(avg_brightness):
    global flash_is_on, flash_pending_state, flash_request_in_flight
    desired_on = avg_brightness < EXTREME_DARK_TH

    with flash_state_lock:
        # Only send when state changes, not on every frame.
        if flash_is_on == desired_on and not flash_request_in_flight:
            return
        if flash_request_in_flight:
            flash_pending_state = desired_on
            return
        if flash_is_on == desired_on:
            return
        flash_is_on = desired_on
        flash_request_in_flight = True

    # Non-blocking HTTP call to avoid slowing inference/video.
    Thread(target=_send_flash_request, args=(desired_on,), daemon=True).start()

# ==================================================
# DB INITIALIZATION
# ==================================================
def create_db_connection():
    global active_db_path, memory_anchor_conn

    def _ensure_memory_anchor():
        global memory_anchor_conn
        if memory_anchor_conn is None:
            memory_anchor_conn = sqlite3.connect(
                DB_MEMORY_URI,
                timeout=10.0,
                check_same_thread=False,
                uri=True,
            )
            memory_anchor_conn.execute("PRAGMA journal_mode=MEMORY")
            memory_anchor_conn.execute("PRAGMA synchronous=OFF")
            memory_anchor_conn.execute("PRAGMA busy_timeout=5000")

    def _try_open(path, uri=False):
        if path == DB_MEMORY_URI:
            _ensure_memory_anchor()
        conn = sqlite3.connect(path, timeout=10.0, check_same_thread=False, uri=uri)
        if path == DB_MEMORY_URI:
            conn.execute("PRAGMA journal_mode=MEMORY")
            conn.execute("PRAGMA synchronous=OFF")
        else:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _cleanup_journal(path):
        if path.startswith("file:"):
            return
        journal_path = f"{path}-journal"
        if os.path.exists(journal_path):
            try:
                os.remove(journal_path)
            except OSError:
                pass

    for path in (active_db_path, DB_FALLBACK_NAME):
        try:
            conn = _try_open(path, uri=path.startswith("file:"))
            if path != active_db_path:
                active_db_path = path
                print(f"Using fallback database path: {active_db_path}")
            return conn
        except sqlite3.Error:
            _cleanup_journal(path)
            try:
                conn = _try_open(path, uri=path.startswith("file:"))
                if path != active_db_path:
                    active_db_path = path
                    print(f"Recovered on fallback database path: {active_db_path}")
                return conn
            except sqlite3.Error:
                continue

    try:
        conn = _try_open(DB_MEMORY_URI, uri=True)
        active_db_path = DB_MEMORY_URI
        print("Using in-memory database fallback.")
        return conn
    except sqlite3.Error as exc:
        raise sqlite3.OperationalError(
            f"Unable to open SQLite database at '{DB_NAME}' or fallback '{DB_FALLBACK_NAME}': {exc}"
        )


def _initialize_runtime_schema(conn):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date TEXT NOT NULL,
            session TEXT NOT NULL DEFAULT 'Daily',
            time TEXT NOT NULL DEFAULT '00:00:00',
            status TEXT NOT NULL DEFAULT 'Absent',
            UNIQUE(student_id, date)
        )
        """
    )
    conn.commit()
    _apply_attendance_schema_migrations(conn)


def _table_exists(cursor, table_name):
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _normalize_attendance_rows(cursor):
    cursor.execute(
        """
        UPDATE attendance
        SET status = CASE
            WHEN LOWER(TRIM(status)) = 'present' THEN ?
            ELSE ?
        END
        """,
        (ATTENDANCE_STATUS_PRESENT, ATTENDANCE_STATUS_ABSENT),
    )
    cursor.execute(
        "UPDATE attendance SET session = ? WHERE session IS NULL OR TRIM(session) = ''",
        (ATTENDANCE_DEFAULT_SESSION,),
    )
    cursor.execute(
        "UPDATE attendance SET time = ? WHERE time IS NULL OR TRIM(time) = ''",
        (ATTENDANCE_DEFAULT_TIME,),
    )


def _deduplicate_attendance_rows(cursor):
    cursor.execute(
        """
        DELETE FROM attendance
        WHERE id NOT IN (
            SELECT COALESCE(
                MAX(CASE WHEN LOWER(TRIM(status)) = 'present' THEN id END),
                MAX(id)
            )
            FROM attendance
            GROUP BY student_id, date
        )
        """
    )


def _deduplicate_attendance_for_date(cursor, date_str):
    cursor.execute(
        """
        DELETE FROM attendance
        WHERE date = ?
          AND id NOT IN (
              SELECT COALESCE(
                  MAX(CASE WHEN LOWER(TRIM(status)) = 'present' THEN id END),
                  MAX(id)
              )
              FROM attendance
              WHERE date = ?
              GROUP BY student_id, date
          )
        """,
        (date_str, date_str),
    )


def _apply_attendance_schema_migrations(conn):
    cursor = conn.cursor()
    cursor.execute("BEGIN IMMEDIATE")
    try:
        _normalize_attendance_rows(cursor)
        _deduplicate_attendance_rows(cursor)
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_attendance_student_date ON attendance(student_id, date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_sid_date ON attendance(student_id, date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_date_sid ON attendance(date, student_id)"
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _is_sunday(now):
    return now.weekday() == 6


def _attendance_date_str(now):
    return now.strftime(ATTENDANCE_DATE_FORMAT)


def _get_all_student_ids(cursor):
    student_ids = set()

    if _table_exists(cursor, "students"):
        cursor.execute("SELECT student_id FROM students WHERE student_id IS NOT NULL")
        for row in cursor.fetchall():
            sid = str(row[0]).strip()
            if sid:
                student_ids.add(sid)

    if not student_ids and os.path.isdir(FACES_DATASET_PATH):
        for entry in os.scandir(FACES_DATASET_PATH):
            if entry.is_dir():
                sid = entry.name.strip()
                if sid:
                    student_ids.add(sid)

    if not student_ids:
        for sid in globals().get("db_identity_names", []):
            sid = str(sid).strip()
            if sid:
                student_ids.add(sid)

    return sorted(student_ids)


def ensure_daily_attendance_initialized(conn=None, now=None):
    global daily_init_cache_key

    now = now or datetime.now()
    day_key = now.strftime("%Y-%m-%d")
    if daily_init_cache_key == day_key:
        return

    with daily_init_lock:
        if daily_init_cache_key == day_key:
            return

        if _is_sunday(now):
            daily_init_cache_key = day_key
            return

        owns_connection = conn is None
        work_conn = conn if conn is not None else create_db_connection()

        try:
            date_str = _attendance_date_str(now)
            cursor = work_conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")

            _deduplicate_attendance_for_date(cursor, date_str)

            inserted_count = 0
            used_students_table = False

            if _table_exists(cursor, "students"):
                before_changes = work_conn.total_changes
                cursor.execute(
                    """
                    INSERT INTO attendance (student_id, date, session, time, status)
                    SELECT TRIM(s.student_id), ?, ?, ?, ?
                    FROM students s
                    WHERE s.student_id IS NOT NULL
                      AND TRIM(s.student_id) <> ''
                      AND NOT EXISTS (
                          SELECT 1
                          FROM attendance a
                          WHERE a.student_id = TRIM(s.student_id)
                            AND a.date = ?
                      )
                    """,
                    (
                        date_str,
                        ATTENDANCE_DEFAULT_SESSION,
                        ATTENDANCE_DEFAULT_TIME,
                        ATTENDANCE_STATUS_ABSENT,
                        date_str,
                    ),
                )
                inserted_count += max(0, work_conn.total_changes - before_changes)

                cursor.execute(
                    "SELECT COUNT(*) FROM students WHERE student_id IS NOT NULL AND TRIM(student_id) <> ''"
                )
                used_students_table = int(cursor.fetchone()[0]) > 0

            if not used_students_table:
                student_ids = _get_all_student_ids(cursor)
                if student_ids:
                    before_changes = work_conn.total_changes
                    cursor.executemany(
                        """
                        INSERT INTO attendance (student_id, date, session, time, status)
                        SELECT ?, ?, ?, ?, ?
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM attendance
                            WHERE student_id = ? AND date = ?
                        )
                        """,
                        [
                            (
                                student_id,
                                date_str,
                                ATTENDANCE_DEFAULT_SESSION,
                                ATTENDANCE_DEFAULT_TIME,
                                ATTENDANCE_STATUS_ABSENT,
                                student_id,
                                date_str,
                            )
                            for student_id in student_ids
                        ],
                    )
                    inserted_count += max(0, work_conn.total_changes - before_changes)

            work_conn.commit()
            daily_init_cache_key = day_key

            if inserted_count > 0:
                print(f"Daily attendance initialized for {date_str}: {inserted_count} ABSENT records.")
        except Exception:
            try:
                work_conn.rollback()
            except Exception:
                pass
            raise
        finally:
            if owns_connection:
                try:
                    work_conn.close()
                except Exception:
                    pass


def init_database(max_retries=3):
    global active_db_path
    last_exc = None

    for attempt in range(max_retries):
        conn = None
        try:
            conn = create_db_connection()
            _initialize_runtime_schema(conn)
            return True
        except sqlite3.Error as exc:
            last_exc = exc
            time.sleep(0.3 * (attempt + 1))
        finally:
            if conn is not None:
                conn.close()

    if active_db_path != DB_MEMORY_URI:
        conn = None
        try:
            active_db_path = DB_MEMORY_URI
            print("Switching runtime database to in-memory mode.")
            conn = create_db_connection()
            _initialize_runtime_schema(conn)
            return True
        except sqlite3.Error as exc:
            last_exc = exc
        finally:
            if conn is not None:
                conn.close()

    print(f"Database initialization failed: {last_exc}")
    return False


runtime_db_ready = init_database()
if runtime_db_ready:
    try:
        ensure_daily_attendance_initialized()
    except Exception as exc:
        print(f"Daily attendance bootstrap failed: {exc}")


def daily_initializer_worker():
    while not stop_event.is_set():
        try:
            ensure_daily_attendance_initialized()
        except Exception as exc:
            print(f"Daily attendance background init failed: {exc}")
        stop_event.wait(30.0)


if runtime_db_ready:
    Thread(target=daily_initializer_worker, daemon=True).start()


attendance_queue = Queue(maxsize=DB_QUEUE_MAX)

# ==================================================
# ONNX SESSION HELPER
# ==================================================
def make_session(model_path):
    available = ort.get_available_providers()
    providers = []
    if "QNNExecutionProvider" in available:
        providers.append(("QNNExecutionProvider", qnn))
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(model_path, providers=providers)

# ==================================================
# LOAD MODELS & DB
# ==================================================
print("⏳ Loading Models...")

face_sess = make_session(FACE_MODEL_PATH)
face_input = face_sess.get_inputs()[0].name

live_sess = make_session(LIVENESS_MODEL_PATH)
live_input_name = live_sess.get_inputs()[0].name
live_input_type = live_sess.get_inputs()[0].type
LIVE_H, LIVE_W = live_sess.get_inputs()[0].shape[2], live_sess.get_inputs()[0].shape[3]

recog_sess = make_session(RECOG_MODEL_PATH)
recog_input = recog_sess.get_inputs()[0].name

def normalize_embedding(emb):
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

def load_face_db(path):
    """
    Supports:
      1) {name: embedding}
      2) {name: [emb1, emb2, ...]}
      3) {name: np.ndarray}
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    names = []
    embeddings = []

    if isinstance(data, dict):
        for name, value in data.items():
            if value is None:
                continue

            # value is a list/tuple of embeddings
            if isinstance(value, (list, tuple)) and len(value) > 0:
                first = value[0]
                if isinstance(first, (list, tuple, np.ndarray)):
                    for emb in value:
                        emb = normalize_embedding(emb)
                        names.append(str(name))
                        embeddings.append(emb)
                else:
                    emb = normalize_embedding(value)
                    names.append(str(name))
                    embeddings.append(emb)
            else:
                emb = normalize_embedding(value)
                names.append(str(name))
                embeddings.append(emb)
    else:
        raise ValueError("FACE_DB_PKL must contain a dictionary.")

    if len(embeddings) == 0:
        return [], np.empty((0, 0), dtype=np.float32)

    return names, np.vstack(embeddings).astype(np.float32)

def build_identity_db(sample_names, sample_embeddings):
    grouped = {}
    for name, emb in zip(sample_names, sample_embeddings):
        grouped.setdefault(name, []).append(emb)

    identity_names = []
    identity_embeddings = []

    for name, emb_list in grouped.items():
        proto = normalize_embedding(np.mean(np.vstack(emb_list), axis=0))
        identity_names.append(name)
        identity_embeddings.append(proto)

    if len(identity_embeddings) == 0:
        return [], np.empty((0, 0), dtype=np.float32)

    return identity_names, np.vstack(identity_embeddings).astype(np.float32)

def calibrate_similarity_thresholds(identity_names, identity_embeddings):
    """
    Uses impostor similarities inside the current DB to set safer open-set thresholds.
    """
    n = identity_embeddings.shape[0]
    if n < 2:
        return SIM_ACCEPT_TH, SIM_UNCERTAIN_TH, SIM_MARGIN_TH, {name: SIM_ACCEPT_TH for name in identity_names}

    sim_matrix = np.dot(identity_embeddings, identity_embeddings.T).astype(np.float32)
    upper = sim_matrix[np.triu_indices(n, k=1)]

    p95 = float(np.percentile(upper, 95))
    p99 = float(np.percentile(upper, 99))
    imp_max = float(np.max(upper))

    accept_th = float(np.clip(max(p99 + 0.012, p95 + 0.03), 0.68, 0.90))
    uncertain_th = float(np.clip(p95 + 0.006, 0.55, accept_th - 0.015))
    margin_th = float(np.clip((imp_max - p95) * 0.5 + 0.015, 0.02, 0.08))

    identity_accept_th = {}
    for i, name in enumerate(identity_names):
        row = sim_matrix[i].copy()
        row[i] = -1.0
        nearest_impostor = float(np.max(row))
        id_th = max(accept_th, nearest_impostor + 0.01)
        identity_accept_th[name] = float(np.clip(id_th, accept_th, 0.93))

    print(
        f"[SIM CALIB] p95={p95:.3f}, p99={p99:.3f}, max={imp_max:.3f} -> "
        f"accept={accept_th:.3f}, uncertain={uncertain_th:.3f}, margin={margin_th:.3f}"
    )

    return accept_th, uncertain_th, margin_th, identity_accept_th

try:
    db_names, db_embeddings = load_face_db(FACE_DB_PKL)
    db_identity_names, db_identity_embeddings = build_identity_db(db_names, db_embeddings)

    if AUTO_CALIBRATE_SIM and len(db_identity_names) > 1:
        SIM_ACCEPT_TH, SIM_UNCERTAIN_TH, SIM_MARGIN_TH, db_identity_accept_th = calibrate_similarity_thresholds(
            db_identity_names,
            db_identity_embeddings,
        )
        # --- CHANGE: Slightly relax calibrated similarity thresholds (without changing calibration formula) ---
        relax_offset = 0.02
        SIM_ACCEPT_TH = max(0.5, SIM_ACCEPT_TH - relax_offset)
        SIM_UNCERTAIN_TH = max(0.45, SIM_UNCERTAIN_TH - relax_offset)
        db_identity_accept_th = {
            name: float(th - relax_offset) for name, th in db_identity_accept_th.items()
        }
        # --- CHANGE: Print final adjusted thresholds ---
        print(
            f"[SIM ADJUST] final_accept={SIM_ACCEPT_TH:.3f}, final_uncertain={SIM_UNCERTAIN_TH:.3f}, "
            f"final_identity_accept={ {k: round(v, 3) for k, v in db_identity_accept_th.items()} }"
        )
    else:
        db_identity_accept_th = {name: SIM_ACCEPT_TH for name in db_identity_names}

    print(
        f"Loaded {len(db_names)} samples, {len(db_identity_names)} identities. "
        f"Thresholds: accept={SIM_ACCEPT_TH:.3f}, uncertain={SIM_UNCERTAIN_TH:.3f}, margin={SIM_MARGIN_TH:.3f}"
    )
    print(f"✅ Loaded {len(db_names)} face embeddings from database.")
except Exception as e:
    print(f"⚠️ Error loading {FACE_DB_PKL}: {e}")
    db_names = []
    db_embeddings = np.empty((0, 0), dtype=np.float32)
    db_identity_names = []
    db_identity_embeddings = np.empty((0, 0), dtype=np.float32)
    db_identity_accept_th = {}

# ==================================================
# HELPERS
# ==================================================
class BoxSmoother:
    def __init__(self, alpha=0.7, reset_dist=60):
        self.alpha = alpha
        self.reset_dist = reset_dist
        self.box = None

    def update(self, b):
        b = np.array(b, dtype=np.float32)
        if self.box is None:
            self.box = b
        else:
            dist = np.hypot(
                (b[0] + b[2] / 2) - (self.box[0] + self.box[2] / 2),
                (b[1] + b[3] / 2) - (self.box[1] + self.box[3] / 2),
            )
            if dist > self.reset_dist:
                self.box = b
            else:
                self.box = self.alpha * b + (1 - self.alpha) * self.box
        return self.box.astype(int)

def blur_score_gray(face_gray):
    return cv2.Laplacian(face_gray, cv2.CV_64F).var()

def preprocess_recog(face):
    face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = (face.astype(np.float32) / 255.0 - 0.5) / 0.5
    return face.transpose(2, 0, 1)[None].astype(np.float32)

def decode_face(hm, bx, fw, fh, avg):
    hm = hm[0, 0].astype(np.float32)
    bx = bx[0].astype(np.float32)

    # Keep your original heuristic
    th = 160 if avg > 90 else 145

    _, max_val, _, max_loc = cv2.minMaxLoc(hm)
    if max_val <= th:
        return None

    x, y = max_loc
    reg = bx[:, y, x]

    cx = (x + 0.5) * (fw / 80)
    cy = (y + 0.5) * (fh / 60)
    bw = (reg[2] / 255) * fw * 2.2
    bh = (reg[3] / 255) * fh * 2.8

    return [int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)], float(max_val / 255.0)

def match_identity(test_emb):
    """
    Returns:
      name, best_similarity, match_state
      match_state in: CONFIDENT, UNCERTAIN, UNKNOWN, NO_DB
    """
    if db_identity_embeddings.size == 0:
        return "UNKNOWN", 0.0, "NO_DB"

    test_emb = normalize_embedding(test_emb)
    sims = np.dot(db_identity_embeddings, test_emb)

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if sims.size > 1:
        tmp = sims.copy()
        tmp[best_idx] = -np.inf
        second_sim = float(np.max(tmp))
    else:
        second_sim = -1.0

    margin = best_sim - second_sim if sims.size > 1 else best_sim
    best_name = db_identity_names[best_idx]
    id_accept_th = db_identity_accept_th.get(best_name, SIM_ACCEPT_TH)

    # Strong accept only when similarity is high enough and clearly wins.
    if best_sim >= id_accept_th and margin >= SIM_MARGIN_TH:
        return best_name, best_sim, "CONFIDENT"

    # Similar but not confident: keep it as uncertain to avoid wrong IDs.
    if best_sim >= SIM_UNCERTAIN_TH:
        return "UNKNOWN", best_sim, "UNCERTAIN"

    return "UNKNOWN", best_sim, "UNKNOWN"

def attendance_writer_worker():
    conn = None
    cursor = None

    while not stop_event.is_set():
        if conn is None:
            try:
                conn = create_db_connection()
                cursor = conn.cursor()
            except Exception as exc:
                print(f"Database reconnect failed: {exc}")
                conn = None
                cursor = None
                time.sleep(0.5)
                continue

        try:
            student_id, date_str, session, time_str, daily_key = attendance_queue.get(timeout=0.2)
        except Empty:
            continue

        try:
            cursor.execute("BEGIN IMMEDIATE")
            _deduplicate_attendance_for_date(cursor, date_str)

            cursor.execute(
                """
                UPDATE attendance
                SET status = ?,
                    session = ?,
                    time = ?
                WHERE student_id = ?
                  AND date = ?
                  AND LOWER(TRIM(status)) <> 'present'
                """,
                (
                    ATTENDANCE_STATUS_PRESENT,
                    session,
                    time_str,
                    student_id,
                    date_str,
                ),
            )

            action = "updated"
            if cursor.rowcount == 0:
                cursor.execute(
                    "SELECT 1 FROM attendance WHERE student_id = ? AND date = ? LIMIT 1",
                    (student_id, date_str),
                )
                exists = cursor.fetchone() is not None

                if not exists:
                    cursor.execute(
                        """
                        INSERT INTO attendance (student_id, date, session, time, status)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            student_id,
                            date_str,
                            session,
                            time_str,
                            ATTENDANCE_STATUS_PRESENT,
                        ),
                    )
                    action = "inserted"
                else:
                    action = "already-present"

            conn.commit()
            if action in ("updated", "inserted"):
                print(f"Marked {student_id} Present: {date_str} ({session} at {time_str})")
        except Exception as exc:
            print(f"Database write error: {exc}")
            with cooldown_lock:
                COOLDOWN_SET.discard(daily_key)
            try:
                conn.close()
            except Exception:
                pass
            conn = None
            cursor = None
        finally:
            attendance_queue.task_done()

    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def _attendance_session_label(now):
    return "Morning" if now.hour < 12 else "Afternoon"


def mark_attendance(student_id):
    global cooldown_day_key

    now = datetime.now()
    if _is_sunday(now):
        return

    date_str = _attendance_date_str(now)
    time_str = now.strftime("%H:%M:%S")
    session = _attendance_session_label(now)

    try:
        ensure_daily_attendance_initialized(now=now)
    except Exception as exc:
        print(f"Daily attendance init failed during mark: {exc}")
        return

    daily_key = f"{student_id}_{date_str}"
    with cooldown_lock:
        if cooldown_day_key != date_str:
            COOLDOWN_SET.clear()
            cooldown_day_key = date_str

        if daily_key in COOLDOWN_SET:
            return
        COOLDOWN_SET.add(daily_key)

    try:
        attendance_queue.put_nowait((student_id, date_str, session, time_str, daily_key))
    except Full:
        with cooldown_lock:
            COOLDOWN_SET.discard(daily_key)
        print("Attendance queue is full; dropping one attendance event.")


Thread(target=attendance_writer_worker, daemon=True).start()
# ==================================================
# CAMERA & STATE
# ==================================================
class Camera:
    def __init__(self):
        print(f"📡 Connecting to ESP32-CAM at {ESP32_CAM_URL} ...")
        self.frame = None
        self.frame_id = 0
        self.frame_lock = Lock()
        self.running = True
        self.stream_thread = Thread(target=self.update, daemon=True)
        self.stream_thread.start()

    def _extract_latest_jpeg(self, buffer):
        # Keep only the newest complete JPEG to avoid decode backlog and latency growth.
        end = buffer.rfind(b"\xff\xd9")
        if end == -1:
            if len(buffer) > ESP32_MAX_BUFFER_BYTES:
                buffer[:] = buffer[-ESP32_CHUNK_SIZE:]
            return None

        start = buffer.rfind(b"\xff\xd8", 0, end)
        if start == -1:
            del buffer[:end + 2]
            return None

        latest_jpg = bytes(buffer[start:end + 2])
        del buffer[:end + 2]

        if len(buffer) > ESP32_MAX_BUFFER_BYTES:
            buffer[:] = buffer[-ESP32_CHUNK_SIZE:]

        return latest_jpg

    def _set_frame(self, frame):
        with self.frame_lock:
            self.frame = frame
            self.frame_id += 1

    def _get_frame(self, copy_frame):
        with self.frame_lock:
            if self.frame is None:
                return None
            return self.frame.copy() if copy_frame else self.frame

    def _get_new_frame(self, last_frame_id, copy_frame):
        with self.frame_lock:
            if self.frame is None or self.frame_id == last_frame_id:
                return None, last_frame_id
            frame = self.frame.copy() if copy_frame else self.frame
            return frame, self.frame_id

    def _stream_loop(self):
        req = urllib.request.Request(
            ESP32_CAM_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        with urllib.request.urlopen(req, timeout=ESP32_STREAM_TIMEOUT) as stream:
            byte_buffer = bytearray()

            while self.running:
                chunk = stream.read(ESP32_CHUNK_SIZE)
                if not chunk:
                    raise ConnectionError("ESP32 stream ended unexpectedly")

                byte_buffer.extend(chunk)
                jpg_bytes = self._extract_latest_jpeg(byte_buffer)
                if jpg_bytes is None:
                    continue

                arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self._set_frame(frame)

    def update(self):
        reconnect_notice_printed = False

        while self.running:
            try:
                self._stream_loop()
                reconnect_notice_printed = False
            except Exception as e:
                if not self.running:
                    break

                if not reconnect_notice_printed:
                    print("⚠️ ESP32 stream lost, reconnecting...")
                    reconnect_notice_printed = True
                else:
                    print(f"⚠️ Reconnect attempt failed: {e}")

                time.sleep(ESP32_RECONNECT_DELAY)

    def read(self, copy_frame=True):
        return self._get_frame(copy_frame=copy_frame)

    def read_if_new(self, last_frame_id, copy_frame=False):
        return self._get_new_frame(last_frame_id=last_frame_id, copy_frame=copy_frame)

    def stop(self):
        self.running = False
        try:
            self.stream_thread.join(timeout=1.0)
        except Exception:
            pass

cam = Camera()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

state_lock = Lock()
inference_state = {
    "box": None,
    "label": "SPOOF",
    "name": "UNKNOWN",
    "reason": "INIT",
    "votes": "0/0",
    "fps_display": 0,
    "low_light": False
}

inference_running = True

# ==================================================
# INFERENCE WORKER
# ==================================================
def inference_worker():
    smoother = BoxSmoother()
    vote_buf = deque(maxlen=VOTE_WINDOW)
    emb_accumulation_buf = deque(maxlen=STRICT_WINDOW)
    fps_buffer = deque(maxlen=15)

    live_votes = 0
    fps_time_sum = 0.0
    consecutive_live = 0
    prev_time = time.time()
    current_fps_to_show = 0

    verification_start = 0.0
    display_lock_until = 0.0
    last_confident_name = None
    consecutive_id_hits = 0

    last_cam_frame_id = -1
    frame_counter = 0
    tracked_box = None
    tracked_box_age = FACE_DET_EVERY_N
    det_input = np.empty((1, 1, 480, 640), dtype=np.uint8)
    emb_sum = None

    live_input_f32 = np.empty((1, 3, LIVE_H, LIVE_W), dtype=np.float32)
    live_input_f16 = np.empty((1, 3, LIVE_H, LIVE_W), dtype=np.float16) if "float16" in live_input_type else None
    inv_127_5 = np.float32(1.0 / 127.5)

    face_run = face_sess.run
    live_run = live_sess.run
    recog_run = recog_sess.run

    def reset_identity_buffer():
        nonlocal emb_sum
        emb_accumulation_buf.clear()
        emb_sum = None

    def reset_votes():
        nonlocal live_votes
        vote_buf.clear()
        live_votes = 0

    def reset_temporal_state():
        nonlocal consecutive_live, last_confident_name, consecutive_id_hits, verification_start
        reset_votes()
        reset_identity_buffer()
        consecutive_live = 0
        last_confident_name = None
        consecutive_id_hits = 0
        verification_start = 0.0

    def append_vote(vote):
        nonlocal live_votes
        if len(vote_buf) == vote_buf.maxlen:
            live_votes -= vote_buf[0]
        vote_buf.append(vote)
        live_votes += vote

    def push_fps_sample(frame_time):
        nonlocal fps_time_sum
        if len(fps_buffer) == fps_buffer.maxlen:
            fps_time_sum -= fps_buffer[0]
        fps_buffer.append(frame_time)
        fps_time_sum += frame_time

    update_lcd("System Ready\nWaiting for Face")

    while inference_running and not stop_event.is_set():
        frame, last_cam_frame_id = cam.read_if_new(last_cam_frame_id, copy_frame=False)
        if frame is None:
            time.sleep(0.002)
            continue

        frame_counter += 1
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = float(cv2.mean(gray)[0])
        update_flash_by_brightness(avg_brightness)
        is_low_light = avg_brightness < LOW_LIGHT_WARN_TH

        blur_th = 60.0 if avg_brightness > 90 else (55.0 if avg_brightness > 70 else 45.0)

        current_box = None
        label = "SPOOF"
        reason = "NO FACE"
        name_show = "UNKNOWN"

        run_detector = (tracked_box is None) or (frame_counter % FACE_DET_EVERY_N == 0)
        if run_detector:
            gray_proc = clahe.apply(gray) if avg_brightness < CLAHE_TH else gray
            cv2.resize(gray_proc, (640, 480), dst=det_input[0, 0], interpolation=cv2.INTER_LINEAR)
            out = face_run(None, {face_input: det_input})
            res = decode_face(out[0], out[1], w, h, avg_brightness)

            if res:
                raw, conf = res
                if conf > 0.6:
                    x, y, bw, bh = smoother.update(raw)
                    x = int(np.clip(x, 0, max(0, w - 1)))
                    y = int(np.clip(y, 0, max(0, h - 1)))
                    bw = int(np.clip(bw, 1, max(1, w - x)))
                    bh = int(np.clip(bh, 1, max(1, h - y)))
                    tracked_box = (x, y, bw, bh)
                    tracked_box_age = 0
                else:
                    tracked_box = None
            else:
                tracked_box = None
        else:
            tracked_box_age += 1
            if tracked_box_age > FACE_DET_EVERY_N:
                tracked_box = None

        if tracked_box is not None:
            x, y, bw, bh = tracked_box
            current_box = (x, y, bw, bh)

            face = frame[y:y + bh, x:x + bw]
            face_gray = gray[y:y + bh, x:x + bw]

            if face.size > 0 and face_gray.size > 0:
                ratio = (bw * bh) / float(w * h)

                live_img = cv2.resize(face, (LIVE_W, LIVE_H), interpolation=cv2.INTER_AREA)
                live_img = cv2.cvtColor(live_img, cv2.COLOR_BGR2RGB)
                live_input_f32[0] = live_img.transpose(2, 0, 1)
                live_input_f32 *= inv_127_5
                live_input_f32 -= 1.0

                live_tensor = live_input_f32
                if live_input_f16 is not None:
                    np.copyto(live_input_f16, live_input_f32, casting="unsafe")
                    live_tensor = live_input_f16

                logits = live_run(None, {live_input_name: live_tensor})[0][0]
                if logits.shape[0] >= 2:
                    diff = float(logits[1] - logits[0])
                    if diff >= 0.0:
                        exp_neg = np.exp(-diff)
                        real_score = float(1.0 / (1.0 + exp_neg))
                    else:
                        exp_pos = np.exp(diff)
                        real_score = float(exp_pos / (1.0 + exp_pos))
                else:
                    real_score = float(logits[0])

                if ratio < MIN_FACE_RATIO:
                    vote, reason = 0, "TOO FAR"
                    consecutive_live = 0
                elif ratio > MAX_FACE_RATIO:
                    vote, reason = 0, "TOO CLOSE"
                    consecutive_live = 0
                elif blur_score_gray(face_gray) < blur_th:
                    vote, reason = 0, "BLUR"
                    consecutive_live = 0
                else:
                    if real_score > LIVENESS_THRESHOLD:
                        vote, reason = 1, "CNN"
                        consecutive_live += 1
                    else:
                        vote, reason = 0, "CNN"
                        consecutive_live = 0

                append_vote(vote)

                if len(vote_buf) == VOTE_WINDOW and live_votes >= VOTE_MIN_LIVE and consecutive_live >= MIN_CONSECUTIVE_LIVE:
                    label = "LIVE"
                else:
                    label = "SPOOF"
                    name_show = "SPOOF"
                    reset_identity_buffer()
                    last_confident_name = None
                    consecutive_id_hits = 0

                if label == "LIVE":
                    side = max(bw, bh) * CROP_PADDING_RATIO
                    cx_pad = x + (bw / 2.0)
                    cy_pad = y + (bh / 2.0)

                    px = max(0, int(cx_pad - side / 2.0))
                    py = max(0, int(cy_pad - side / 2.0))
                    side = int(side)

                    if px + side > w:
                        side = w - px
                    if py + side > h:
                        side = h - py

                    if side > 0:
                        padded_face = frame[py:py + side, px:px + side]

                        if padded_face.size > 0:
                            raw_emb = recog_run(None, {recog_input: preprocess_recog(padded_face)})[0][0]
                            raw_emb = normalize_embedding(raw_emb)

                            if len(emb_accumulation_buf) == emb_accumulation_buf.maxlen:
                                oldest = emb_accumulation_buf.popleft()
                                emb_sum -= oldest

                            emb_accumulation_buf.append(raw_emb)
                            if emb_sum is None:
                                emb_sum = raw_emb.copy()
                            else:
                                emb_sum += raw_emb

                    if len(emb_accumulation_buf) >= STRICT_WINDOW and emb_sum is not None:
                        mean_emb = normalize_embedding(emb_sum / len(emb_accumulation_buf))

                        if len(db_identity_names) > 0:
                            name, best_sim, match_state = match_identity(mean_emb)

                            if match_state == "CONFIDENT":
                                if name == last_confident_name:
                                    consecutive_id_hits += 1
                                else:
                                    last_confident_name = name
                                    consecutive_id_hits = 1

                                if consecutive_id_hits >= ID_CONFIRM_FRAMES:
                                    name_show = name
                                else:
                                    name_show = f"COLLECTING ID {consecutive_id_hits}/{ID_CONFIRM_FRAMES}"
                            else:
                                name_show = "UNKNOWN"
                                last_confident_name = None
                                consecutive_id_hits = 0
                        else:
                            name_show = "UNKNOWN"
                    else:
                        name_show = f"COLLECTING {len(emb_accumulation_buf)}/{STRICT_WINDOW}"
            else:
                reset_temporal_state()
        else:
            reset_temporal_state()

        now = time.time()

        if now >= display_lock_until:
            if current_box is not None:
                if verification_start == 0.0:
                    verification_start = now

                elapsed = now - verification_start

                if elapsed < 1.0:
                    update_lcd("Verifying...\nPlease Hold")
                else:
                    if label == "SPOOF":
                        update_lcd("Spoof Detected\nAccess Denied")
                        display_lock_until = now + 2.0
                        verification_start = 0.0

                    elif label == "LIVE":
                        name_text = str(name_show)
                        if name_text in ("UNKNOWN", "UNCERTAIN") or "COLLECTING" in name_text:
                            if "COLLECTING" in name_text:
                                update_lcd("Almost there...\nHold Still")
                            else:
                                update_lcd("Unknown Person\nTry Again")
                            display_lock_until = now + 2.0
                            verification_start = 0.0
                        else:
                            short_name = name_text[-4:] if len(name_text) >= 4 else name_text
                            update_lcd(f"Attendance Done\nID: {short_name}")
                            mark_attendance(name_text)

                            display_lock_until = now + 2.0
                            verification_start = 0.0
            else:
                verification_start = 0.0
                update_lcd("System Ready\nWaiting for Face")

        if prev_time > 0:
            frame_time = now - prev_time
            if frame_time > 0:
                push_fps_sample(frame_time)

        prev_time = now

        if len(fps_buffer) > 0 and fps_time_sum > 0:
            avg_frame_time = fps_time_sum / len(fps_buffer)
            current_fps_to_show = int(1.0 / avg_frame_time) if avg_frame_time > 0 else 0
        else:
            current_fps_to_show = 0

        with state_lock:
            inference_state.update({
                "box": current_box,
                "label": label,
                "name": name_show,
                "reason": reason,
                "votes": f"{live_votes}/{len(vote_buf)}",
                "fps_display": current_fps_to_show,
                "low_light": is_low_light,
            })
Thread(target=inference_worker, daemon=True).start()

# ==================================================
# GUI
# ==================================================
root = tk.Tk()
root.title("EdgeID Attendance System")
lbl = tk.Label(root, bg="black")
lbl.pack()

def on_close():
    global inference_running
    inference_running = False
    stop_event.set()
    try:
        cam.stop()
    except Exception:
        pass

    try:
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
    except Exception:
        pass

    root.destroy()

def update_gui():
    if not inference_running:
        return

    frame = cam.read(copy_frame=True)
    if frame is not None:
        with state_lock:
            s = inference_state.copy()

        if s["box"]:
            x, y, bw, bh = s["box"]

            if s["label"] == "LIVE":
                col = (0, 255, 0)
                ui_text = f"LIVE: {s['name']} | {s['votes']} | {s['reason']}"
            else:
                col = (0, 0, 255)
                ui_text = f"WARNING: SPOOF | {s['votes']} | {s['reason']}"

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 2)
            cv2.putText(frame, ui_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        if s["low_light"]:
            cv2.putText(frame, "LOW LIGHT", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"FPS:{s['fps_display']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if frame.shape[1] > DISPLAY_MAX_WIDTH:
            scale = DISPLAY_MAX_WIDTH / float(frame.shape[1])
            disp_h = max(1, int(frame.shape[0] * scale))
            frame = cv2.resize(frame, (DISPLAY_MAX_WIDTH, disp_h), interpolation=cv2.INTER_AREA)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        lbl.configure(image=img)
        lbl.image = img

    root.after(GUI_REFRESH_MS, update_gui)

root.protocol("WM_DELETE_WINDOW", on_close)
update_gui()
root.mainloop()



