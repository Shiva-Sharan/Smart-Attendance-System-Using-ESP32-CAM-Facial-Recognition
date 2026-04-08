import sqlite3

def create_database():
    # Connect to DB (creates file if not exists)
    conn = sqlite3.connect("attendance.db")
    cur = conn.cursor()

    # -----------------------------
    # CREATE STUDENTS TABLE
    # -----------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        password TEXT
    )
    """)

    # -----------------------------
    # CREATE ATTENDANCE TABLE
    # -----------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        date TEXT,
        session TEXT,
        time TEXT,
        status TEXT,
        UNIQUE(student_id, date, session)
    )
    """)

    # -----------------------------
    # INSERT STUDENTS (ID = password)
    # -----------------------------
    students = [
        "160722735067", "160722735071", "160722735072", "160722735076",
        "160722735077", "160722735078", "160722735079", "160722735082",
        "160722735083", "160722735084", "160722735085", "160722735086",
        "160722735093", "160722735094", "160722735097", "1607227350100"
    ]

    for sid in students:
        cur.execute(
            "INSERT OR IGNORE INTO students VALUES (?, ?)",
            (sid, sid)
        )

    conn.commit()
    conn.close()

    print("✅ attendance.db created successfully")
    print("✅ students and attendance tables verified")
    print("✅ students list inserted")

if __name__ == "__main__":
    create_database()