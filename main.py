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
ESP32_CAM_URL = "http://192.168.1.11:8080/video" # <-- REPLACE WITH YOUR ESP32 IP ADDRESS
ESP32_STREAM_TIMEOUT = 8.0
ESP32_RECONNECT_DELAY = 1.0
ESP32_CHUNK_SIZE = 4096
ESP32_MAX_BUFFER_BYTES = 2 * 1024 * 1024
FACE_DET_EVERY_N = 2
# GUI runs lighter to leave more CPU budget to inference.
GUI_REFRESH_MS = 60
DISPLAY_MAX_WIDTH = 960
MQTT_RECONNECT_MIN_SEC = 1
MQTT_RECONNECT_MAX_SEC = 8
MQTT_PUBLISH_QOS = 1
DB_QUEUE_MAX = 256

# --- PERFORMANCE TUNING ---
ENABLE_RUNTIME_DEBUG_LOGS = False
ENABLE_LIGHTWEIGHT_PROFILING = False
# Reduced logging cadence to lower print/I-O overhead in the hot loop.
PERF_LOG_INTERVAL_SEC = 4.0
NPU_TIMING_LOG_INTERVAL_SEC = 4.0
NPU_INFERENCE_WARN_MS = 8.0
# Slower adaptive cadence + stability hits prevents stride oscillation.
ADAPTIVE_TUNING_INTERVAL_SEC = 1.2
ADAPT_STABLE_HITS = 2
ADAPT_SEVERE_LOW_DELTA = 2
ADAPT_SEVERE_HIGH_DELTA = 3
ADAPT_FPS_LOW = 25
ADAPT_FPS_HIGH = 30
DET_EVERY_N_MIN = max(1, FACE_DET_EVERY_N)
DET_EVERY_N_MAX = 4
LIVENESS_EVERY_N_MIN = 1
LIVENESS_EVERY_N_MAX = 4
RECOG_EVERY_N_MIN = 1
RECOG_EVERY_N_MAX = 3
# Push less frequently to GUI state dict to reduce lock contention.
STATE_PUBLISH_EVERY_N = 2
LIVENESS_CACHE_MAX_FRAMES = 3
LIVENESS_CACHE_MAX_REUSE_FRAMES = 6
# Reuse liveness decision when tiny face motion is detected.
LIVENESS_MOTION_THUMB_SIZE = (24, 24)
LIVENESS_MOTION_REUSE_DIFF_TH = 2.2
LIVENESS_SKIP_AFTER_CONSEC_LIVE = 5
LIVENESS_STABLE_SKIP_FRAMES = 3
# Reuse recognition embedding when identity is stable and face motion is minimal.
RECOG_MOTION_THUMB_SIZE = (20, 20)
RECOG_MOTION_REUSE_DIFF_TH = 1.8
RECOG_CACHE_MAX_REUSE_FRAMES = 6
ID_STABLE_EXTRA_RECOG_SKIP = 1
RECOG_SKIP_AFTER_ID_HITS = 5
RECOG_STABLE_SKIP_FRAMES = 3
# Re-evaluate blur on a cadence instead of every frame.
BLUR_CHECK_EVERY_N = 3
BRIGHTNESS_UPDATE_EVERY_N = 2
FACE_STABLE_TRACK_AGE_MAX = 2
FACE_STABLE_MOTION_PX = 4.0
# Keep inference at source resolution (ESP32 QVGA) to preserve detail and avoid wasteful upscale compute.
PROCESS_AT_SOURCE_RESOLUTION = True
# Keep detector/crop behavior consistent with enrollment pipeline.
FACE_DETECTOR_BACKEND = "onnx"  # auto|haar|onnx
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_FACE_PX = 28
HAAR_MAX_CANDIDATES = 6
# Flash control logic is not evaluated every frame to reduce CPU and HTTP churn.
FLASH_UPDATE_EVERY_N = 3

# --- THRESHOLDS ---
# Fallback values; runtime values are calibrated from FACE_DB_PKL when possible.
SIM_ACCEPT_TH = 0.40        # fallback confident accept cutoff
SIM_UNCERTAIN_TH = 0.62    # fallback uncertain cutoff
SIM_MARGIN_TH = 0.035       # best must beat second-best by this margin
AUTO_CALIBRATE_SIM = True
SIM_CALIBRATION_RELAX_OFFSET = 0.14
SIM_ACCEPT_MIN = 0.54
SIM_UNCERTAIN_MIN = 0.42
SIM_ID_ACCEPT_SPREAD = 0.06
ID_CONFIRM_FRAMES = 3       # require same confident identity this many times
ENABLE_MATCH_SCORE_LOGS = False
MATCH_LOG_INTERVAL_SEC = 0.5

CROP_PADDING_RATIO = 1.25
LIVENESS_THRESHOLD = 0.65
LIVENESS_THRESHOLD_LOW_LIGHT = 0.60
LIVENESS_THRESHOLD_FLASH = 0.58
LIVENESS_THRESHOLD_NORMAL = LIVENESS_THRESHOLD

MIN_FACE_RATIO = 0.12
MAX_FACE_RATIO = 0.45

VOTE_WINDOW = 12
VOTE_MIN_LIVE = 10
MIN_CONSECUTIVE_LIVE = 8

STRICT_WINDOW = 6

BASE_BLUR = 60.0
LOW_LIGHT_WARN_TH = 55
# Flash controller thresholds and timing (tuned to avoid ON/OFF oscillation).
LOW_LIGHT_THRESHOLD = 45.0
FLASH_OFF_THRESHOLD = 55.0
LIGHT_STATE_SMOOTH_ALPHA = 0.16
LIGHT_STATE_LOW_ENTER = 52.0
LIGHT_STATE_LOW_EXIT = 62.0
LIGHT_SPIKE_DELTA = 14.0
LIGHT_STATE_TRANSITION_SEC = 0.45
FLASH_LIVENESS_STABILIZE_SEC = 1.10
FLASH_ON_VALUE = 255
FLASH_OFF_VALUE = 0
FLASH_SMOOTH_ALPHA = 0.20
FLASH_MIN_ON_SEC = 2.5
FLASH_MIN_OFF_SEC = 1.0
FLASH_SWITCH_COOLDOWN_SEC = 0.8
FLASH_SETTLE_ON_SEC = 1.2
FLASH_SETTLE_OFF_SEC = 0.8
FLASH_DARK_DEBOUNCE_FRAMES = 5
FLASH_BRIGHT_DEBOUNCE_FRAMES = 6
FLASH_HTTP_TIMEOUT_SEC = 0.25
FLASH_DEBUG_LOG_INTERVAL_SEC = 1.0
FLASH_DECISION_LOG_INTERVAL_SEC = 0.4
LIVENESS_LOG_INTERVAL_SEC = 1.0
FACE_TIGHT_CROP_RATIO = 0.84
FACE_DARK_MEAN_TH = 22.0
FACE_BRIGHT_MEAN_TH = 225.0
UNDEREXPOSED_PIXEL_TH = 20
OVEREXPOSED_PIXEL_TH = 245
FACE_DARK_RATIO_TH = 0.35
FACE_BRIGHT_RATIO_TH = 0.22
CLAHE_TH = 70
# Run expensive CLAHE only when the face is extremely dark.
CLAHE_EXTREME_TH = 58.0
# Equalize under flash only when highlights are strongly elevated.
FLASH_EQUALIZE_FACE_MEAN_TH = 95.0

qnn = {
    "backend_path": "QnnHtp.dll",
    "htp_performance_mode": "high_performance",
    "rpc_control_latency": "low",
}
ORT_AVAILABLE_PROVIDERS = tuple(ort.get_available_providers())
ORT_QNN_AVAILABLE = "QNNExecutionProvider" in ORT_AVAILABLE_PROVIDERS
COOLDOWN_SET = set()
cooldown_lock = Lock()
stop_event = Event()
active_db_path = DB_NAME
memory_anchor_conn = None
daily_init_lock = Lock()
daily_init_cache_key = None
cooldown_day_key = None

cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, min(4, (os.cpu_count() or 1))))


def runtime_debug_log(message):
    if ENABLE_RUNTIME_DEBUG_LOGS:
        print(message)

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
flash_is_on = False
flash_pending_state = None
flash_request_in_progress = False
last_on_time = 0.0
last_off_time = 0.0
last_switch_time = 0.0
ignore_brightness_until = 0.0
flash_brightness_ema = None
dark_frame_streak = 0
bright_frame_streak = 0
last_flash_debug_log = 0.0
last_flash_decision_log = 0.0


def _flash_log_decision(message, now=None):
    global last_flash_decision_log
    if not ENABLE_RUNTIME_DEBUG_LOGS:
        return
    if now is None:
        now = time.time()
    if (now - last_flash_decision_log) >= FLASH_DECISION_LOG_INTERVAL_SEC:
        print(message)
        last_flash_decision_log = now


def _apply_flash_transition_locked(next_state, now):
    global flash_is_on, flash_request_in_progress
    global last_on_time, last_off_time, last_switch_time, ignore_brightness_until
    global dark_frame_streak, bright_frame_streak

    flash_is_on = next_state
    flash_request_in_progress = True
    last_switch_time = now

    if next_state:
        last_on_time = now
        ignore_brightness_until = now + FLASH_SETTLE_ON_SEC
        bright_frame_streak = 0
    else:
        last_off_time = now
        ignore_brightness_until = now + FLASH_SETTLE_OFF_SEC
        dark_frame_streak = 0


def send_flash_request(turn_on: bool):
    global flash_request_in_progress, flash_pending_state
    global flash_is_on, last_switch_time
    if not ESP32_FLASH_BASE_URL:
        with flash_state_lock:
            flash_request_in_progress = False
        _flash_log_decision("[FLASH] control URL unavailable; skipping command")
        return

    val = FLASH_ON_VALUE if turn_on else FLASH_OFF_VALUE
    url = f"{ESP32_FLASH_BASE_URL}/control?var=led_intensity&val={val}"
    try:
        with urllib.request.urlopen(url, timeout=FLASH_HTTP_TIMEOUT_SEC):
            pass
        runtime_debug_log(f"[FLASH] {'ON' if turn_on else 'OFF'} sent (val={val})")
    except Exception as exc:
        runtime_debug_log(f"[FLASH] request failed for {'ON' if turn_on else 'OFF'}: {exc}")
    finally:
        next_state = None
        now = time.time()
        with flash_state_lock:
            flash_request_in_progress = False
            if flash_pending_state is not None and flash_pending_state != flash_is_on:
                candidate_state = flash_pending_state
                flash_pending_state = None

                cooldown_left = FLASH_SWITCH_COOLDOWN_SEC - (now - last_switch_time)
                if cooldown_left > 0:
                    flash_pending_state = candidate_state
                    _flash_log_decision(
                        f"[FLASH] pending {'ON' if candidate_state else 'OFF'} delayed by cooldown ({cooldown_left:.2f}s)",
                        now=now,
                    )
                else:
                    _apply_flash_transition_locked(candidate_state, now)
                    next_state = candidate_state
            else:
                flash_pending_state = None

        if next_state is not None:
            runtime_debug_log(f"[FLASH] transition -> {'ON' if next_state else 'OFF'} (queued)")
            Thread(target=send_flash_request, args=(next_state,), daemon=True).start()


def update_flash_by_brightness(avg_brightness):
    global flash_is_on, flash_pending_state, flash_request_in_progress
    global flash_brightness_ema, dark_frame_streak, bright_frame_streak
    global last_flash_debug_log, last_on_time, last_off_time, last_switch_time
    global ignore_brightness_until

    now = time.time()
    raw_brightness = float(avg_brightness)
    transition_state = None
    transition_raw = 0.0
    transition_filtered = 0.0

    snapshot = None
    with flash_state_lock:
        if flash_brightness_ema is None:
            flash_brightness_ema = raw_brightness
        else:
            flash_brightness_ema = (
                (1.0 - FLASH_SMOOTH_ALPHA) * flash_brightness_ema
                + FLASH_SMOOTH_ALPHA * raw_brightness
            )

        filtered_brightness = float(flash_brightness_ema)

        if (now - last_flash_debug_log) >= FLASH_DEBUG_LOG_INTERVAL_SEC:
            hold_left = max(0.0, ignore_brightness_until - now)
            runtime_debug_log(
                f"[FLASH DBG] raw={raw_brightness:.1f} filt={filtered_brightness:.1f} "
                f"state={'ON' if flash_is_on else 'OFF'} dark={dark_frame_streak} bright={bright_frame_streak} "
                f"hold={hold_left:.2f}s req={'Y' if flash_request_in_progress else 'N'}"
            )
            last_flash_debug_log = now

        # Ignore immediate post-switch brightness to avoid flash/self-exposure feedback loops.
        if now < ignore_brightness_until:
            return flash_is_on, ignore_brightness_until, last_switch_time

        if filtered_brightness < LOW_LIGHT_THRESHOLD:
            dark_frame_streak += 1
            bright_frame_streak = 0
        elif filtered_brightness > FLASH_OFF_THRESHOLD:
            bright_frame_streak += 1
            dark_frame_streak = 0
        else:
            dark_frame_streak = 0
            bright_frame_streak = 0

        desired_state = flash_is_on

        if flash_is_on:
            on_elapsed = now - last_on_time
            if bright_frame_streak >= FLASH_BRIGHT_DEBOUNCE_FRAMES:
                if on_elapsed >= FLASH_MIN_ON_SEC:
                    desired_state = False
                else:
                    _flash_log_decision(
                        f"[FLASH] keep ON (min-on {FLASH_MIN_ON_SEC:.1f}s, elapsed={on_elapsed:.2f}s)",
                        now=now,
                    )
        else:
            off_elapsed = now - last_off_time
            if dark_frame_streak >= FLASH_DARK_DEBOUNCE_FRAMES:
                if off_elapsed >= FLASH_MIN_OFF_SEC:
                    desired_state = True
                else:
                    _flash_log_decision(
                        f"[FLASH] keep OFF (min-off {FLASH_MIN_OFF_SEC:.1f}s, elapsed={off_elapsed:.2f}s)",
                        now=now,
                    )

        if desired_state == flash_is_on:
            return flash_is_on, ignore_brightness_until, last_switch_time

        since_switch = now - last_switch_time
        if since_switch < FLASH_SWITCH_COOLDOWN_SEC:
            _flash_log_decision(
                f"[FLASH] switch blocked by cooldown ({FLASH_SWITCH_COOLDOWN_SEC - since_switch:.2f}s left)",
                now=now,
            )
            return flash_is_on, ignore_brightness_until, last_switch_time

        if flash_request_in_progress:
            flash_pending_state = desired_state
            _flash_log_decision(
                f"[FLASH] queued {'ON' if desired_state else 'OFF'} while request in progress",
                now=now,
            )
            return flash_is_on, ignore_brightness_until, last_switch_time

        _apply_flash_transition_locked(desired_state, now)
        transition_state = desired_state
        transition_raw = raw_brightness
        transition_filtered = filtered_brightness
        snapshot = (flash_is_on, ignore_brightness_until, last_switch_time)

    if transition_state is not None:
        runtime_debug_log(
            f"[FLASH] transition -> {'ON' if transition_state else 'OFF'} "
            f"raw={transition_raw:.1f} filt={transition_filtered:.1f}"
        )
        Thread(target=send_flash_request, args=(transition_state,), daemon=True).start()

    if snapshot is None:
        snapshot = get_flash_controller_snapshot()
    return snapshot


def get_flash_controller_snapshot():
    with flash_state_lock:
        return flash_is_on, ignore_brightness_until, last_switch_time

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
def _build_ort_fallback_chain():
    if ORT_QNN_AVAILABLE:
        return [("QNNExecutionProvider", qnn), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


ORT_FALLBACK_PROVIDER_CHAIN = _build_ort_fallback_chain()


def _log_session_debug(session):
    print("Available providers:", ort.get_available_providers())
    print("Session providers:", session.get_providers())
    input_type = session.get_inputs()[0].type
    print("Model input type:", input_type)
    if input_type == "tensor(float)":
        print("WARNING: Model is float32 → may not run on NPU efficiently")


def make_session(model_path, model_name):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Keep execution deterministic and avoid CPU oversubscription.
    sess_options.intra_op_num_threads = 3 if (os.cpu_count() or 1) >= 6 else 2
    sess_options.inter_op_num_threads = 2
    sess_options.enable_mem_pattern = True
    sess_options.enable_mem_reuse = True
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")

    # QNN execution test: try NPU-only first (no CPU in provider chain).
    if ORT_QNN_AVAILABLE:
        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=[("QNNExecutionProvider", qnn)],
            )
            _log_session_debug(session)
            return session
        except Exception as exc:
            print(f"QNN failed → falling back to CPU ({model_name}): {exc}")
    else:
        print(f"QNN failed → falling back to CPU ({model_name}): provider not available")

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=ORT_FALLBACK_PROVIDER_CHAIN,
    )
    _log_session_debug(session)
    return session

# ==================================================
# LOAD MODELS & DB
# ==================================================
print("⏳ Loading Models...")

print(f"[ORT] Available providers: {list(ORT_AVAILABLE_PROVIDERS)}")
if ORT_QNN_AVAILABLE:
    print("[ORT] QNNExecutionProvider detected and configured as primary provider.")
else:
    print("[ORT] QNNExecutionProvider not found. Running with CPU fallback only.")

face_sess = None
face_input = None
face_providers = []
haar_face_cascade = None
face_detector_mode = "onnx"

if FACE_DETECTOR_BACKEND in ("auto", "haar"):
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        haar_face_cascade = cv2.CascadeClassifier(cascade_path)
        if haar_face_cascade.empty():
            raise RuntimeError(f"Failed to load cascade: {cascade_path}")
        face_detector_mode = "haar"
        print(f"[DETECT] Using Haar detector at source resolution ({cascade_path})")
    except Exception as e:
        haar_face_cascade = None
        if FACE_DETECTOR_BACKEND == "haar":
            raise
        print(f"[DETECT] Haar unavailable, falling back to ONNX detector: {e}")

if face_detector_mode != "haar":
    face_sess = make_session(FACE_MODEL_PATH, "face")
    face_input = face_sess.get_inputs()[0].name
    face_providers = face_sess.get_providers()
    print(f"[ORT] Face session providers: {face_providers}")
else:
    print("[DETECT] ONNX face detector bypassed to avoid full-frame upscaling.")

live_sess = make_session(LIVENESS_MODEL_PATH, "liveness")
live_input_name = live_sess.get_inputs()[0].name
live_input_type = live_sess.get_inputs()[0].type
LIVE_H, LIVE_W = live_sess.get_inputs()[0].shape[2], live_sess.get_inputs()[0].shape[3]
live_providers = live_sess.get_providers()
print(f"[ORT] Liveness session providers: {live_providers}")

recog_sess = make_session(RECOG_MODEL_PATH, "recognition")
recog_input = recog_sess.get_inputs()[0].name
recog_input_type = recog_sess.get_inputs()[0].type
recog_providers = recog_sess.get_providers()
print(f"[ORT] Recognition session providers: {recog_providers}")

qnn_provider_groups = [live_providers, recog_providers] + ([face_providers] if face_providers else [])
qnn_primary_all = all(
    providers and providers[0] == "QNNExecutionProvider"
    for providers in qnn_provider_groups
)
qnn_listed_all = all(
    "QNNExecutionProvider" in providers
    for providers in qnn_provider_groups
)
print(f"[ORT] NPU primary across all sessions: {'YES' if qnn_primary_all else 'NO'}")
print(f"[ORT] NPU listed in all sessions: {'YES' if qnn_listed_all else 'NO'}")

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
        # Relax calibration to improve recall on enrollment-consistent crops.
        SIM_ACCEPT_TH = float(np.clip(SIM_ACCEPT_TH - SIM_CALIBRATION_RELAX_OFFSET, SIM_ACCEPT_MIN, 0.90))
        SIM_UNCERTAIN_TH = float(
            np.clip(
                SIM_UNCERTAIN_TH - SIM_CALIBRATION_RELAX_OFFSET,
                SIM_UNCERTAIN_MIN,
                max(SIM_UNCERTAIN_MIN, SIM_ACCEPT_TH - 0.02),
            )
        )
        id_accept_floor = max(SIM_UNCERTAIN_TH + 0.01, SIM_ACCEPT_TH - 0.03)
        id_accept_ceil = SIM_ACCEPT_TH + SIM_ID_ACCEPT_SPREAD
        db_identity_accept_th = {
            name: float(np.clip(th - SIM_CALIBRATION_RELAX_OFFSET, id_accept_floor, id_accept_ceil))
            for name, th in db_identity_accept_th.items()
        }
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
    return cv2.Laplacian(face_gray, cv2.CV_32F).var()

def preprocess_recog(face):
    face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)
    # Use a single float conversion pass and in-place normalization.
    face = face[:, :, ::-1].astype(np.float32)  # BGR -> RGB
    face *= np.float32(1.0 / 127.5)
    face -= 1.0
    return face.transpose(2, 0, 1)[None]


def preprocess_liveness_face(face_bgr, lighting_state, clahe_obj, roi_weight, face_mean=None, face_dark_ratio=None):
    face = cv2.resize(face_bgr, (LIVE_W, LIVE_H), interpolation=cv2.INTER_AREA)

    # Fast path for normal/mild lighting: skip YCrCb + histogram operations.
    use_clahe = (
        lighting_state == "LOW_LIGHT"
        and (
            face_mean is None
            or face_mean < CLAHE_EXTREME_TH
            or (face_dark_ratio is not None and face_dark_ratio >= (FACE_DARK_RATIO_TH + 0.10))
        )
    )
    use_flash_equalize = (
        lighting_state == "FLASH_ACTIVE"
        and (face_mean is None or face_mean >= FLASH_EQUALIZE_FACE_MEAN_TH)
    )

    if use_clahe or use_flash_equalize:
        ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        y_chan = ycrcb[:, :, 0]

        if use_clahe:
            y_chan = clahe_obj.apply(y_chan)
        else:
            np.minimum(y_chan, 235, out=y_chan)
            y_chan = cv2.equalizeHist(y_chan)

        ycrcb[:, :, 0] = y_chan
        face_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    cv2.multiply(face_rgb, roi_weight, dst=face_rgb, scale=1.0 / 255.0)
    return face_rgb

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
            final_name, candidate_name, best_similarity, second_similarity, margin, id_accept_threshold, match_state
      match_state in: CONFIDENT, UNCERTAIN, UNKNOWN, NO_DB
    """
    if db_identity_embeddings.size == 0:
                return "UNKNOWN", "UNKNOWN", 0.0, -1.0, 0.0, SIM_ACCEPT_TH, "NO_DB"

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
        return best_name, best_name, best_sim, second_sim, margin, id_accept_th, "CONFIDENT"

    # Similar but not confident: keep it as uncertain to avoid wrong IDs.
    if best_sim >= SIM_UNCERTAIN_TH:
        return "UNKNOWN", best_name, best_sim, second_sim, margin, id_accept_th, "UNCERTAIN"

    return "UNKNOWN", best_name, best_sim, second_sim, margin, id_accept_th, "UNKNOWN"

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
                    # Keep ESP32 native resolution (QVGA) to avoid detail loss from upscaling.
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

    detector_stride = 2
    # Start slightly lighter; adaptive tuning can still tighten when FPS is high.
    liveness_stride = min(2, LIVENESS_EVERY_N_MAX)
    recog_stride = min(2, RECOG_EVERY_N_MAX)
    last_adapt_ts = time.time()
    adapt_low_hits = 0
    adapt_high_hits = 0

    last_cam_frame_id = -1
    frame_counter = 0
    tracked_box = None
    tracked_box_age = FACE_DET_EVERY_N
    prev_tracked_box_for_stride = None
    use_haar_detector = (face_detector_mode == "haar" and haar_face_cascade is not None)
    det_input = None
    det_input_view = None
    det_h, det_w = 0, 0
    face_feed = None
    if not use_haar_detector:
        det_input = np.empty((1, 1, 480, 640), dtype=np.uint8)
        det_input_view = det_input[0, 0]
        det_h, det_w = det_input_view.shape
        face_feed = {face_input: det_input}
    emb_sum = None

    live_input_f32 = np.empty((1, 3, LIVE_H, LIVE_W), dtype=np.float32)
    live_input_f16 = np.empty((1, 3, LIVE_H, LIVE_W), dtype=np.float16) if "float16" in live_input_type else None
    live_feed_f32 = {live_input_name: live_input_f32}
    live_feed_f16 = {live_input_name: live_input_f16} if live_input_f16 is not None else None
    inv_127_5 = np.float32(1.0 / 127.5)
    live_thumb_curr = np.empty((LIVENESS_MOTION_THUMB_SIZE[1], LIVENESS_MOTION_THUMB_SIZE[0]), dtype=np.uint8)
    live_thumb_prev = np.empty_like(live_thumb_curr)
    live_thumb_valid = False

    recog_resize_bgr = np.empty((112, 112, 3), dtype=np.uint8)
    recog_resize_rgb = np.empty((112, 112, 3), dtype=np.uint8)
    recog_norm_hwc = np.empty((112, 112, 3), dtype=np.float32)
    recog_input_f32 = np.empty((1, 3, 112, 112), dtype=np.float32)
    recog_input_f16 = np.empty((1, 3, 112, 112), dtype=np.float16) if "float16" in recog_input_type else None
    recog_feed_f32 = {recog_input: recog_input_f32}
    recog_feed_f16 = {recog_input: recog_input_f16} if recog_input_f16 is not None else None
    recog_thumb_curr = np.empty((RECOG_MOTION_THUMB_SIZE[1], RECOG_MOTION_THUMB_SIZE[0]), dtype=np.uint8)
    recog_thumb_prev = np.empty_like(recog_thumb_curr)
    recog_thumb_valid = False
    last_recog_eval_frame = -1000
    recog_skip_until = -1
    last_blur_eval_frame = -1000
    last_blur_score = 0.0
    liveness_skip_until = -1
    flash_on_now = False
    flash_brightness_hold_until = 0.0
    avg_brightness = 0.0
    last_avg_brightness = 0.0
    detector_min_face_px = HAAR_MIN_FACE_PX

    liveness_roi_weight = np.full((LIVE_H, LIVE_W), 96, dtype=np.uint8)
    cv2.ellipse(
        liveness_roi_weight,
        (LIVE_W // 2, LIVE_H // 2),
        (int(LIVE_W * 0.40), int(LIVE_H * 0.47)),
        0,
        0,
        360,
        255,
        -1,
    )
    liveness_roi_weight_3c = cv2.merge([liveness_roi_weight, liveness_roi_weight, liveness_roi_weight])

    scene_brightness_ema = None
    lighting_state = "NORMAL_LIGHT"
    lighting_stable_until = 0.0
    flash_liveness_ignore_until = 0.0
    last_liveness_log_ts = 0.0

    last_live_vote = None
    last_live_score = 0.0
    last_live_eval_frame = -1000
    last_match_log_ts = 0.0

    perf_last_log_ts = time.time()
    perf_frames = 0
    perf_t_gray = 0.0
    perf_t_flash = 0.0
    perf_t_detect = 0.0
    perf_t_live = 0.0
    perf_t_recog = 0.0
    npu_timing_last_log_ts = time.time()
    face_infer_time_sum = 0.0
    face_infer_count = 0
    live_infer_time_sum = 0.0
    live_infer_count = 0
    recog_infer_time_sum = 0.0
    recog_infer_count = 0

    face_run = face_sess.run if face_sess is not None else None
    live_run = live_sess.run
    recog_run = recog_sess.run

    def reset_identity_buffer():
        nonlocal emb_sum, last_recog_eval_frame, recog_thumb_valid, recog_skip_until
        emb_accumulation_buf.clear()
        emb_sum = None
        last_recog_eval_frame = -1000
        recog_thumb_valid = False
        recog_skip_until = -1

    def reset_votes():
        nonlocal live_votes
        vote_buf.clear()
        live_votes = 0

    def reset_temporal_state():
        nonlocal consecutive_live, last_confident_name, consecutive_id_hits, verification_start
        nonlocal last_live_vote, last_live_score, last_live_eval_frame, live_thumb_valid
        nonlocal last_blur_eval_frame, last_blur_score, liveness_skip_until
        reset_votes()
        reset_identity_buffer()
        consecutive_live = 0
        last_confident_name = None
        consecutive_id_hits = 0
        verification_start = 0.0
        last_live_vote = None
        last_live_score = 0.0
        last_live_eval_frame = -1000
        live_thumb_valid = False
        last_blur_eval_frame = -1000
        last_blur_score = 0.0
        liveness_skip_until = -1

    def append_vote(vote):
        nonlocal live_votes
        if vote is None:
            return
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

    def prepare_recog_feed(face_bgr):
        # Reuse pre-allocated buffers to avoid per-frame tensor allocations for recognition.
        cv2.resize(face_bgr, (112, 112), dst=recog_resize_bgr, interpolation=cv2.INTER_AREA)
        cv2.cvtColor(recog_resize_bgr, cv2.COLOR_BGR2RGB, dst=recog_resize_rgb)
        np.multiply(recog_resize_rgb, inv_127_5, out=recog_norm_hwc, casting="unsafe")
        np.subtract(recog_norm_hwc, 1.0, out=recog_norm_hwc)
        recog_input_f32[0] = recog_norm_hwc.transpose(2, 0, 1)
        if recog_input_f16 is not None:
            np.copyto(recog_input_f16, recog_input_f32, casting="unsafe")
            return recog_feed_f16
        return recog_feed_f32

    update_lcd("System Ready\nWaiting for Face")
    frame_h = -1
    frame_w = -1
    frame_area_inv = 0.0

    while inference_running and not stop_event.is_set():
        frame, last_cam_frame_id = cam.read_if_new(last_cam_frame_id, copy_frame=False)
        if frame is None or frame.size == 0:
            time.sleep(0.002)
            continue

        frame_counter += 1
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            time.sleep(0.002)
            continue
        if h != frame_h or w != frame_w:
            frame_h, frame_w = h, w
            frame_area_inv = 1.0 / float(max(1, w * h))
            detector_min_face_px = max(HAAR_MIN_FACE_PX, int(min(h, w) * 0.08))

        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_counter == 1 or (frame_counter % BRIGHTNESS_UPDATE_EVERY_N == 0):
            avg_brightness = float(gray.mean(dtype=np.float32))
            last_avg_brightness = avg_brightness
        else:
            avg_brightness = last_avg_brightness
        perf_t_gray += time.perf_counter() - t0

        t0 = time.perf_counter()
        # Evaluate flash less frequently to reduce lock churn and avoid needless HTTP transitions.
        if frame_counter == 1 or (frame_counter % FLASH_UPDATE_EVERY_N == 0):
            flash_on_now, flash_brightness_hold_until, _ = update_flash_by_brightness(avg_brightness)
        perf_t_flash += time.perf_counter() - t0

        frame_now = time.time()

        if scene_brightness_ema is None:
            prev_scene_ema = avg_brightness
            scene_brightness_ema = avg_brightness
        else:
            prev_scene_ema = scene_brightness_ema
            scene_brightness_ema = (
                (1.0 - LIGHT_STATE_SMOOTH_ALPHA) * scene_brightness_ema
                + LIGHT_STATE_SMOOTH_ALPHA * avg_brightness
            )

        brightness_spike = abs(avg_brightness - prev_scene_ema) >= LIGHT_SPIKE_DELTA

        next_lighting_state = lighting_state
        if flash_on_now or frame_now < flash_brightness_hold_until or (brightness_spike and avg_brightness > scene_brightness_ema):
            next_lighting_state = "FLASH_ACTIVE"
        elif scene_brightness_ema <= LIGHT_STATE_LOW_ENTER:
            next_lighting_state = "LOW_LIGHT"
        elif scene_brightness_ema >= LIGHT_STATE_LOW_EXIT:
            next_lighting_state = "NORMAL_LIGHT"
        elif lighting_state == "FLASH_ACTIVE":
            next_lighting_state = "LOW_LIGHT" if scene_brightness_ema < LOW_LIGHT_WARN_TH else "NORMAL_LIGHT"

        if next_lighting_state != lighting_state:
            runtime_debug_log(
                f"[LIGHT] {lighting_state} -> {next_lighting_state} "
                f"raw={avg_brightness:.1f} ema={scene_brightness_ema:.1f} spike={int(brightness_spike)}"
            )
            lighting_state = next_lighting_state
            lighting_stable_until = frame_now + LIGHT_STATE_TRANSITION_SEC
            if lighting_state == "FLASH_ACTIVE":
                flash_liveness_ignore_until = frame_now + FLASH_LIVENESS_STABILIZE_SEC
            else:
                flash_liveness_ignore_until = max(flash_liveness_ignore_until, frame_now + 0.2)
            reset_temporal_state()

        if ENABLE_RUNTIME_DEBUG_LOGS and (frame_now - last_liveness_log_ts) >= LIVENESS_LOG_INTERVAL_SEC:
            hold_left = max(0.0, max(lighting_stable_until, flash_liveness_ignore_until) - frame_now)
            runtime_debug_log(
                f"[LIGHT DBG] raw={avg_brightness:.1f} ema={scene_brightness_ema:.1f} "
                f"state={lighting_state} flash={int(flash_on_now)} hold={hold_left:.2f}s"
            )
            last_liveness_log_ts = frame_now

        is_low_light = lighting_state == "LOW_LIGHT" or scene_brightness_ema < LOW_LIGHT_WARN_TH

        if lighting_state == "LOW_LIGHT":
            blur_th = 42.0
            live_threshold = LIVENESS_THRESHOLD_LOW_LIGHT
        elif lighting_state == "FLASH_ACTIVE":
            blur_th = 62.0
            live_threshold = LIVENESS_THRESHOLD_FLASH
        else:
            blur_th = 55.0 if scene_brightness_ema > 70 else 48.0
            live_threshold = LIVENESS_THRESHOLD_NORMAL

        current_box = None
        label = "SPOOF"
        reason = "NO FACE"
        name_show = "UNKNOWN"

        det_t0 = time.perf_counter()
        track_motion_px = float("inf")
        if tracked_box is not None and prev_tracked_box_for_stride is not None:
            tcx = tracked_box[0] + (tracked_box[2] * 0.5)
            tcy = tracked_box[1] + (tracked_box[3] * 0.5)
            pcx = prev_tracked_box_for_stride[0] + (prev_tracked_box_for_stride[2] * 0.5)
            pcy = prev_tracked_box_for_stride[1] + (prev_tracked_box_for_stride[3] * 0.5)
            track_motion_px = float(np.hypot(tcx - pcx, tcy - pcy))
        face_stable_for_det = (
            tracked_box is not None
            and tracked_box_age <= FACE_STABLE_TRACK_AGE_MAX
            and track_motion_px <= FACE_STABLE_MOTION_PX
        )
        # Adaptive detector cadence: stable face -> lighter detector rate, movement -> faster refresh.
        detector_stride = 4 if face_stable_for_det else 2
        run_detector = (tracked_box is None) or (frame_counter % detector_stride == 0)
        if run_detector:
            # CLAHE for detector is reserved for truly dark frames.
            use_clahe_for_det = lighting_state == "LOW_LIGHT" and avg_brightness < CLAHE_EXTREME_TH
            gray_proc = clahe.apply(gray) if use_clahe_for_det else gray
            if use_haar_detector:
                # Native-resolution detection path (no full-frame upscaling).
                faces = haar_face_cascade.detectMultiScale(
                    gray_proc,
                    scaleFactor=HAAR_SCALE_FACTOR,
                    minNeighbors=HAAR_MIN_NEIGHBORS,
                    flags=cv2.CASCADE_SCALE_IMAGE,
                    minSize=(detector_min_face_px, detector_min_face_px),
                )
                if len(faces) > 0:
                    if len(faces) > HAAR_MAX_CANDIDATES:
                        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[:HAAR_MAX_CANDIDATES]
                    raw = max(faces, key=lambda b: b[2] * b[3])
                    x, y, bw, bh = smoother.update(raw)
                    x = int(np.clip(x, 0, max(0, w - 1)))
                    y = int(np.clip(y, 0, max(0, h - 1)))
                    bw = int(np.clip(bw, 1, max(1, w - x)))
                    bh = int(np.clip(bh, 1, max(1, h - y)))
                    tracked_box = (x, y, bw, bh)
                    tracked_box_age = 0
                else:
                    tracked_box = None
            elif face_run is not None and face_feed is not None:
                # ONNX fallback path (kept for compatibility if native detector cannot load).
                if gray_proc.shape[0] == det_h and gray_proc.shape[1] == det_w:
                    np.copyto(det_input_view, gray_proc)
                else:
                    cv2.resize(gray_proc, (det_w, det_h), dst=det_input_view, interpolation=cv2.INTER_LINEAR)
                infer_t0 = time.perf_counter()
                out = face_run(None, face_feed)
                face_infer_time_sum += time.perf_counter() - infer_t0
                face_infer_count += 1
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
                tracked_box = None
        else:
            tracked_box_age += 1
            if tracked_box_age > detector_stride:
                tracked_box = None
        if tracked_box is not None:
            prev_tracked_box_for_stride = tracked_box
        else:
            prev_tracked_box_for_stride = None
        perf_t_detect += time.perf_counter() - det_t0

        if tracked_box is not None:
            x, y, bw, bh = tracked_box
            current_box = (x, y, bw, bh)

            if bw > 0 and bh > 0:
                ratio = (bw * bh) * frame_area_inv

                tight_w = max(1, int(bw * FACE_TIGHT_CROP_RATIO))
                tight_h = max(1, int(bh * FACE_TIGHT_CROP_RATIO))
                tight_x = int(np.clip(x + (bw - tight_w) // 2, 0, max(0, w - tight_w)))
                tight_y = int(np.clip(y + (bh - tight_h) // 2, 0, max(0, h - tight_h)))

                face_tight = frame[tight_y:tight_y + tight_h, tight_x:tight_x + tight_w]
                face_gray_tight = gray[tight_y:tight_y + tight_h, tight_x:tight_x + tight_w]

                vote = None

                if face_tight.size == 0 or face_gray_tight.size == 0:
                    reason = "FACE ROI"
                    consecutive_live = 0
                    last_live_vote = None
                    live_thumb_valid = False
                elif frame_now < lighting_stable_until:
                    reason = "STABILIZE_LIGHT"
                    consecutive_live = 0
                    last_live_vote = None
                    live_thumb_valid = False
                elif frame_now < flash_liveness_ignore_until:
                    reason = "STABILIZE_FLASH"
                    consecutive_live = 0
                    last_live_vote = None
                    live_thumb_valid = False
                elif ratio < MIN_FACE_RATIO:
                    reason = "TOO FAR"
                    consecutive_live = 0
                    last_live_vote = None
                    live_thumb_valid = False
                elif ratio > MAX_FACE_RATIO:
                    reason = "TOO CLOSE"
                    consecutive_live = 0
                    last_live_vote = None
                    live_thumb_valid = False
                else:
                    if (
                        last_blur_eval_frame < 0
                        or (frame_counter - last_blur_eval_frame) >= BLUR_CHECK_EVERY_N
                    ):
                        last_blur_score = blur_score_gray(face_gray_tight)
                        last_blur_eval_frame = frame_counter

                    if last_blur_score < blur_th:
                        reason = "BLUR"
                        consecutive_live = 0
                        last_live_vote = None
                        live_thumb_valid = False
                    else:
                        face_mean = float(cv2.mean(face_gray_tight)[0])
                        dark_ratio = float(np.count_nonzero(face_gray_tight <= UNDEREXPOSED_PIXEL_TH)) / float(face_gray_tight.size)
                        bright_ratio = float(np.count_nonzero(face_gray_tight >= OVEREXPOSED_PIXEL_TH)) / float(face_gray_tight.size)

                        if face_mean <= FACE_DARK_MEAN_TH or dark_ratio >= FACE_DARK_RATIO_TH:
                            reason = "UNDEREXPOSED"
                            consecutive_live = 0
                            last_live_vote = None
                            live_thumb_valid = False
                        elif face_mean >= FACE_BRIGHT_MEAN_TH or bright_ratio >= FACE_BRIGHT_RATIO_TH:
                            reason = "OVEREXPOSED"
                            consecutive_live = 0
                            last_live_vote = None
                            live_thumb_valid = False
                        else:
                            cache_age = frame_counter - last_live_eval_frame
                            stable_live_skip = (
                                last_live_vote is not None
                                and last_live_vote > 0
                                and consecutive_live > LIVENESS_SKIP_AFTER_CONSEC_LIVE
                                and frame_counter <= liveness_skip_until
                            )
                            should_run_liveness = (
                                not stable_live_skip
                                and (
                                    last_live_vote is None
                                    or (frame_counter % liveness_stride == 0)
                                    or (cache_age > LIVENESS_CACHE_MAX_FRAMES)
                                )
                            )
                            reuse_cached_liveness = False
                            thumb_computed = False

                            # Skip liveness inference when the face ROI barely changed.
                            if (
                                should_run_liveness
                                and last_live_vote is not None
                                and cache_age <= LIVENESS_CACHE_MAX_REUSE_FRAMES
                            ):
                                cv2.resize(
                                    face_gray_tight,
                                    LIVENESS_MOTION_THUMB_SIZE,
                                    dst=live_thumb_curr,
                                    interpolation=cv2.INTER_AREA,
                                )
                                thumb_computed = True
                                if live_thumb_valid:
                                    motion_delta = cv2.norm(live_thumb_curr, live_thumb_prev, cv2.NORM_L1) / float(live_thumb_curr.size)
                                    if motion_delta <= LIVENESS_MOTION_REUSE_DIFF_TH:
                                        reuse_cached_liveness = True

                            if should_run_liveness and not reuse_cached_liveness:
                                live_t0 = time.perf_counter()
                                live_img = preprocess_liveness_face(
                                    face_tight,
                                    lighting_state,
                                    clahe,
                                    liveness_roi_weight_3c,
                                    face_mean=face_mean,
                                    face_dark_ratio=dark_ratio,
                                )

                                live_input_f32[0] = live_img.transpose(2, 0, 1)
                                live_input_f32 *= inv_127_5
                                live_input_f32 -= 1.0

                                infer_t0 = time.perf_counter()
                                if live_input_f16 is not None:
                                    np.copyto(live_input_f16, live_input_f32, casting="unsafe")
                                    logits = live_run(None, live_feed_f16)[0][0]
                                else:
                                    logits = live_run(None, live_feed_f32)[0][0]
                                live_infer_time_sum += time.perf_counter() - infer_t0
                                live_infer_count += 1
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

                                if real_score > live_threshold:
                                    vote = 1
                                    reason = f"CNN {lighting_state} {real_score:.2f}"
                                    consecutive_live += 1
                                    if consecutive_live > LIVENESS_SKIP_AFTER_CONSEC_LIVE:
                                        liveness_skip_until = frame_counter + LIVENESS_STABLE_SKIP_FRAMES
                                else:
                                    vote = 0
                                    reason = f"CNN {lighting_state} {real_score:.2f}"
                                    consecutive_live = 0
                                    liveness_skip_until = -1

                                last_live_vote = vote
                                last_live_score = real_score
                                last_live_eval_frame = frame_counter
                                if not thumb_computed:
                                    cv2.resize(
                                        face_gray_tight,
                                        LIVENESS_MOTION_THUMB_SIZE,
                                        dst=live_thumb_curr,
                                        interpolation=cv2.INTER_AREA,
                                    )
                                np.copyto(live_thumb_prev, live_thumb_curr)
                                live_thumb_valid = True
                                perf_t_live += time.perf_counter() - live_t0
                            else:
                                vote = last_live_vote
                                if vote is not None:
                                    reason = f"CNN CACHE {lighting_state} {last_live_score:.2f}"
                                    if vote > 0:
                                        consecutive_live += 1
                                        if consecutive_live > LIVENESS_SKIP_AFTER_CONSEC_LIVE:
                                            liveness_skip_until = frame_counter + LIVENESS_STABLE_SKIP_FRAMES
                                    else:
                                        consecutive_live = 0
                                        liveness_skip_until = -1
                                    if thumb_computed:
                                        np.copyto(live_thumb_prev, live_thumb_curr)
                                        live_thumb_valid = True

                append_vote(vote)

                if len(vote_buf) == VOTE_WINDOW and live_votes >= VOTE_MIN_LIVE and consecutive_live >= MIN_CONSECUTIVE_LIVE:
                    label = "LIVE"
                else:
                    label = "SPOOF"
                    name_show = "SPOOF" if reason.startswith("CNN") else "UNKNOWN"
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
                            stable_id_skip = (
                                last_confident_name is not None
                                and consecutive_id_hits > RECOG_SKIP_AFTER_ID_HITS
                                and frame_counter <= recog_skip_until
                            )
                            should_run_recog = (
                                len(emb_accumulation_buf) < 2
                                or (frame_counter % recog_stride == 0)
                            ) and (not stable_id_skip)
                            if (
                                should_run_recog
                                and last_confident_name is not None
                                and (frame_counter - last_recog_eval_frame) <= (RECOG_CACHE_MAX_REUSE_FRAMES + (recog_stride * ID_STABLE_EXTRA_RECOG_SKIP))
                            ):
                                cv2.resize(
                                    face_gray_tight,
                                    RECOG_MOTION_THUMB_SIZE,
                                    dst=recog_thumb_curr,
                                    interpolation=cv2.INTER_AREA,
                                )
                                if recog_thumb_valid:
                                    recog_motion_delta = cv2.norm(
                                        recog_thumb_curr,
                                        recog_thumb_prev,
                                        cv2.NORM_L1,
                                    ) / float(recog_thumb_curr.size)
                                    if recog_motion_delta <= RECOG_MOTION_REUSE_DIFF_TH:
                                        # Stable-ID + low motion path: reuse previous embedding window.
                                        should_run_recog = False

                            if should_run_recog:
                                rec_t0 = time.perf_counter()
                                infer_t0 = time.perf_counter()
                                raw_emb = recog_run(None, prepare_recog_feed(padded_face))[0][0]
                                recog_infer_time_sum += time.perf_counter() - infer_t0
                                recog_infer_count += 1
                                raw_emb = normalize_embedding(raw_emb)

                                if len(emb_accumulation_buf) == emb_accumulation_buf.maxlen:
                                    oldest = emb_accumulation_buf.popleft()
                                    emb_sum -= oldest

                                emb_accumulation_buf.append(raw_emb)
                                if emb_sum is None:
                                    emb_sum = raw_emb.copy()
                                else:
                                    emb_sum += raw_emb
                                cv2.resize(
                                    face_gray_tight,
                                    RECOG_MOTION_THUMB_SIZE,
                                    dst=recog_thumb_prev,
                                    interpolation=cv2.INTER_AREA,
                                )
                                recog_thumb_valid = True
                                last_recog_eval_frame = frame_counter
                                perf_t_recog += time.perf_counter() - rec_t0

                    if len(emb_accumulation_buf) >= STRICT_WINDOW and emb_sum is not None:
                        mean_emb = normalize_embedding(emb_sum / len(emb_accumulation_buf))

                        if len(db_identity_names) > 0:
                            (
                                name,
                                best_name,
                                best_sim,
                                second_sim,
                                sim_margin,
                                id_accept_th,
                                match_state,
                            ) = match_identity(mean_emb)

                            if ENABLE_MATCH_SCORE_LOGS and (frame_now - last_match_log_ts) >= MATCH_LOG_INTERVAL_SEC:
                                print(
                                    f"[MATCH] state={match_state} cand={best_name} sim={best_sim:.3f} "
                                    f"second={second_sim:.3f} margin={sim_margin:.3f} "
                                    f"accept={id_accept_th:.3f} uncertain={SIM_UNCERTAIN_TH:.3f}"
                                )
                                last_match_log_ts = frame_now

                            if match_state == "CONFIDENT":
                                if name == last_confident_name:
                                    consecutive_id_hits += 1
                                else:
                                    last_confident_name = name
                                    consecutive_id_hits = 1
                                if consecutive_id_hits > RECOG_SKIP_AFTER_ID_HITS:
                                    recog_skip_until = frame_counter + RECOG_STABLE_SKIP_FRAMES

                                if consecutive_id_hits >= ID_CONFIRM_FRAMES:
                                    name_show = name
                                else:
                                    name_show = f"COLLECTING ID {consecutive_id_hits}/{ID_CONFIRM_FRAMES}"
                            else:
                                name_show = "UNKNOWN"
                                last_confident_name = None
                                consecutive_id_hits = 0
                                recog_skip_until = -1
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
                        if reason.startswith("STABILIZE") or reason in ("OVEREXPOSED", "UNDEREXPOSED", "FACE ROI"):
                            update_lcd("Adjust Lighting\nPlease Hold")
                        elif reason in ("BLUR", "TOO FAR", "TOO CLOSE"):
                            update_lcd("Hold Still\nFace Centered")
                        else:
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

        if (now - last_adapt_ts) >= ADAPTIVE_TUNING_INTERVAL_SEC:
            prev_strides = (liveness_stride, recog_stride)
            if current_fps_to_show > 0:
                if current_fps_to_show < ADAPT_FPS_LOW:
                    adapt_low_hits += 1
                    adapt_high_hits = 0
                    low_hits_needed = 1 if current_fps_to_show <= (ADAPT_FPS_LOW - ADAPT_SEVERE_LOW_DELTA) else ADAPT_STABLE_HITS
                    if adapt_low_hits >= low_hits_needed:
                        liveness_stride = min(LIVENESS_EVERY_N_MAX, liveness_stride + 1)
                        recog_stride = min(RECOG_EVERY_N_MAX, recog_stride + 1)
                        adapt_low_hits = 0
                elif current_fps_to_show > ADAPT_FPS_HIGH:
                    adapt_high_hits += 1
                    adapt_low_hits = 0
                    high_hits_needed = 1 if current_fps_to_show >= (ADAPT_FPS_HIGH + ADAPT_SEVERE_HIGH_DELTA) else ADAPT_STABLE_HITS
                    if adapt_high_hits >= high_hits_needed:
                        liveness_stride = max(LIVENESS_EVERY_N_MIN, liveness_stride - 1)
                        recog_stride = max(RECOG_EVERY_N_MIN, recog_stride - 1)
                        adapt_high_hits = 0
                else:
                    adapt_low_hits = 0
                    adapt_high_hits = 0
            else:
                adapt_low_hits = 0
                adapt_high_hits = 0

            if prev_strides != (liveness_stride, recog_stride):
                runtime_debug_log(
                    f"[ADAPT] fps={current_fps_to_show} detN={detector_stride} "
                    f"liveN={liveness_stride} recogN={recog_stride}"
                )
            last_adapt_ts = now

        perf_frames += 1
        if (now - npu_timing_last_log_ts) >= NPU_TIMING_LOG_INTERVAL_SEC:
            face_avg_ms = (face_infer_time_sum * 1000.0 / face_infer_count) if face_infer_count else 0.0
            live_avg_ms = (live_infer_time_sum * 1000.0 / live_infer_count) if live_infer_count else 0.0
            recog_avg_ms = (recog_infer_time_sum * 1000.0 / recog_infer_count) if recog_infer_count else 0.0
            print(
                f"[NPU CHECK] face={face_avg_ms:.2f}ms live={live_avg_ms:.2f}ms recog={recog_avg_ms:.2f}ms"
            )
            if max(face_avg_ms, live_avg_ms, recog_avg_ms) >= NPU_INFERENCE_WARN_MS:
                print("If inference time is high → likely CPU fallback")

            npu_timing_last_log_ts = now
            face_infer_time_sum = 0.0
            face_infer_count = 0
            live_infer_time_sum = 0.0
            live_infer_count = 0
            recog_infer_time_sum = 0.0
            recog_infer_count = 0

        if ENABLE_LIGHTWEIGHT_PROFILING and (now - perf_last_log_ts) >= PERF_LOG_INTERVAL_SEC:
            elapsed = max(1e-6, now - perf_last_log_ts)
            n = max(1, perf_frames)
            print(
                f"[PERF] fps={current_fps_to_show} detN={detector_stride} liveN={liveness_stride} recN={recog_stride} "
                f"gray={perf_t_gray / n * 1000.0:.2f}ms flash={perf_t_flash / n * 1000.0:.2f}ms "
                f"det={perf_t_detect / n * 1000.0:.2f}ms live={perf_t_live / n * 1000.0:.2f}ms "
                f"rec={perf_t_recog / n * 1000.0:.2f}ms loop={n / elapsed:.1f}Hz"
            )
            perf_last_log_ts = now
            perf_frames = 0
            perf_t_gray = 0.0
            perf_t_flash = 0.0
            perf_t_detect = 0.0
            perf_t_live = 0.0
            perf_t_recog = 0.0

        if frame_counter % STATE_PUBLISH_EVERY_N == 0 or current_box is None:
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
last_gui_frame_id = -1

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
    global last_gui_frame_id
    if not inference_running:
        return

    frame, last_gui_frame_id = cam.read_if_new(last_gui_frame_id, copy_frame=True)
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
