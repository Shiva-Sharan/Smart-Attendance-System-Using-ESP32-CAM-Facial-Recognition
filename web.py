import os
import sqlite3
import time
from datetime import datetime
from threading import Lock
from flask import Flask, g, jsonify, redirect, render_template, request, session
from werkzeug.security import check_password_hash

DB_FILE = r"D:\Major Project\attendance.db"
DB_FALLBACK_FILE = r"D:\Major Project\attendance_runtime.db"
DB_MEMORY_URI = "file:edgeid_web_runtime?mode=memory&cache=shared"
ATTENDANCE_DATE_FORMAT = "%d %b %Y"
ATTENDANCE_DEFAULT_SESSION = "Daily"
ATTENDANCE_DEFAULT_TIME = "00:00:00"
ATTENDANCE_STATUS_ABSENT = "Absent"
ATTENDANCE_STATUS_PRESENT = "Present"

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("EDGEID_SECRET_KEY", "Shiva123!"),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
)

_db_init_lock = Lock()
_db_ready = False
_active_db_path = DB_FILE
_memory_anchor_conn = None
_daily_init_lock = Lock()
_daily_init_cache_key = None


def _new_connection():
    global _active_db_path, _memory_anchor_conn

    def _ensure_memory_anchor():
        global _memory_anchor_conn
        if _memory_anchor_conn is None:
            _memory_anchor_conn = sqlite3.connect(
                DB_MEMORY_URI,
                timeout=10.0,
                uri=True,
                check_same_thread=False,
            )
            _memory_anchor_conn.execute("PRAGMA journal_mode=MEMORY")
            _memory_anchor_conn.execute("PRAGMA synchronous=OFF")
            _memory_anchor_conn.execute("PRAGMA busy_timeout=5000")

    def _try_open(path, uri=False):
        if path == DB_MEMORY_URI:
            _ensure_memory_anchor()
        conn = sqlite3.connect(path, timeout=10.0, uri=uri)
        conn.row_factory = sqlite3.Row
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

    for path in (_active_db_path, DB_FALLBACK_FILE):
        try:
            conn = _try_open(path, uri=path.startswith("file:"))
            if path != _active_db_path:
                _active_db_path = path
                print(f"Using fallback database path: {_active_db_path}")
            return conn
        except sqlite3.Error:
            _cleanup_journal(path)
            try:
                conn = _try_open(path, uri=path.startswith("file:"))
                if path != _active_db_path:
                    _active_db_path = path
                    print(f"Recovered on fallback database path: {_active_db_path}")
                return conn
            except sqlite3.Error:
                continue

    try:
        conn = _try_open(DB_MEMORY_URI, uri=True)
        _active_db_path = DB_MEMORY_URI
        print("Using in-memory database fallback.")
        return conn
    except sqlite3.Error as exc:
        raise sqlite3.OperationalError(
            f"Unable to open SQLite database at '{DB_FILE}' or fallback '{DB_FALLBACK_FILE}': {exc}"
        )


def _initialize_schema(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
        """
    )
    conn.execute(
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


def _normalize_attendance_rows(conn):
    conn.execute(
        """
        UPDATE attendance
        SET status = CASE
            WHEN LOWER(TRIM(status)) = 'present' THEN ?
            ELSE ?
        END
        """,
        (ATTENDANCE_STATUS_PRESENT, ATTENDANCE_STATUS_ABSENT),
    )
    conn.execute(
        "UPDATE attendance SET session = ? WHERE session IS NULL OR TRIM(session) = ''",
        (ATTENDANCE_DEFAULT_SESSION,),
    )
    conn.execute(
        "UPDATE attendance SET time = ? WHERE time IS NULL OR TRIM(time) = ''",
        (ATTENDANCE_DEFAULT_TIME,),
    )


def _deduplicate_attendance_rows(conn):
    conn.execute(
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


def _apply_attendance_schema_migrations(conn):
    conn.execute("BEGIN IMMEDIATE")
    try:
        _normalize_attendance_rows(conn)
        _deduplicate_attendance_rows(conn)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_attendance_student_date ON attendance(student_id, date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_sid_date ON attendance(student_id, date)"
        )
        conn.execute(
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


def ensure_daily_attendance_initialized(conn, now=None):
    global _daily_init_cache_key

    now = now or datetime.now()
    day_key = now.strftime("%Y-%m-%d")
    if _daily_init_cache_key == day_key:
        return

    with _daily_init_lock:
        if _daily_init_cache_key == day_key:
            return

        if _is_sunday(now):
            _daily_init_cache_key = day_key
            return

        date_str = _attendance_date_str(now)

        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
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
            conn.commit()
            _daily_init_cache_key = day_key
        except Exception:
            conn.rollback()
            raise


def init_db(max_retries: int = 3) -> bool:
    global _active_db_path
    last_exc = None

    for attempt in range(max_retries):
        conn = None
        try:
            conn = _new_connection()
            _initialize_schema(conn)
            ensure_daily_attendance_initialized(conn)
            return True
        except sqlite3.Error as exc:
            last_exc = exc
            time.sleep(0.3 * (attempt + 1))
        finally:
            if conn is not None:
                conn.close()

    if _active_db_path != DB_MEMORY_URI:
        conn = None
        try:
            _active_db_path = DB_MEMORY_URI
            print("Switching database to in-memory mode.")
            conn = _new_connection()
            _initialize_schema(conn)
            ensure_daily_attendance_initialized(conn)
            return True
        except sqlite3.Error as exc:
            last_exc = exc
        finally:
            if conn is not None:
                conn.close()

    print(f"Database initialization failed: {last_exc}")
    return False


def ensure_db_ready():
    global _db_ready
    if _db_ready:
        return

    with _db_init_lock:
        if _db_ready:
            return
        _db_ready = init_db()


def get_db():
    ensure_db_ready()
    if "db" not in g:
        g.db = _new_connection()
    ensure_daily_attendance_initialized(g.db)
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def _verify_password(stored_password: str, provided_password: str) -> bool:
    # Supports both legacy plain text and hashed values.
    if stored_password == provided_password:
        return True
    if stored_password.startswith(("pbkdf2:", "scrypt:")):
        try:
            return check_password_hash(stored_password, provided_password)
        except Exception:
            return False
    return False


@app.route("/")
def home():
    # Keep existing single-page behavior intact.
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    if "sid" not in session:
        return redirect("/")
    return render_template("dashboard.html", sid=session["sid"])


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    sid = str(data.get("sid", "")).strip()
    pwd = str(data.get("pwd", ""))

    if not sid or not pwd:
        return jsonify({"success": False, "error": "Missing credentials"}), 400

    cur = get_db().cursor()
    cur.execute("SELECT student_id, password FROM students WHERE student_id = ? LIMIT 1", (sid,))
    user = cur.fetchone()

    if user and _verify_password(user["password"], pwd):
        session["sid"] = user["student_id"]
        return jsonify({"success": True})

    return jsonify({"success": False, "error": "Invalid ID or Password"}), 401


@app.route("/attendance-data")
def attendance_data():
    sid = session.get("sid")
    if not sid:
        return jsonify([])

    cur = get_db().cursor()
    cur.execute(
        """
        SELECT date, session, time, status
        FROM attendance
        WHERE student_id = ?
        ORDER BY id DESC
        """,
        (sid,),
    )

    rows = cur.fetchall()
    return jsonify(
        [
            {
                "date": row["date"],
                "session": row["session"],
                "time": row["time"],
                "status": row["status"],
                "valid": 1,
            }
            for row in rows
        ]
    )


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/health")
def health():
    try:
        ensure_db_ready()
        get_db().execute("SELECT 1")
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


ensure_db_ready()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
