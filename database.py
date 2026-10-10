import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
from datetime import datetime, time as dt_time, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.environ.get("FRAS_DB_PATH", os.path.join(BASE_DIR, "face_recognition.db"))
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_ADMINS = (("admin", "admin"), ("ADMIN", "ADMIN"))
HASH_PREFIX = "pbkdf2_sha256"
HASH_ITERATIONS = 200000

SCHEMA = """
CREATE TABLE IF NOT EXISTS admin (
    admin_id TEXT NOT NULL PRIMARY KEY,
    password TEXT NOT NULL,
    created_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT NOT NULL PRIMARY KEY,
    department TEXT,
    face_image_path TEXT NOT NULL,
    created_by TEXT NOT NULL REFERENCES admin(admin_id)
);
CREATE TABLE IF NOT EXISTS attendance (
    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_date_time TEXT NOT NULL,
    exit_date_time TEXT,
    user_id TEXT NOT NULL REFERENCES users(user_id)
);
CREATE INDEX IF NOT EXISTS idx_attendance_user ON attendance(user_id, exit_date_time);
CREATE INDEX IF NOT EXISTS idx_attendance_entry ON attendance(entry_date_time);
"""

_connection = None
_lock = threading.RLock()


class DatabaseError(Exception):
    pass


def get_connection():
    global _connection
    with _lock:
        if _connection is None:
            try:
                directory = os.path.dirname(DATABASE_PATH)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                connection = sqlite3.connect(DATABASE_PATH, timeout=10, isolation_level=None, check_same_thread=False)
                connection.execute("PRAGMA foreign_keys = ON")
            except (sqlite3.Error, OSError) as error:
                raise DatabaseError(f"Unable to open the database file ({DATABASE_PATH}): {error}") from error
            _connection = connection
        return _connection


def close_connection():
    global _connection
    with _lock:
        if _connection is not None:
            try:
                _connection.close()
            except sqlite3.Error:
                pass
        _connection = None


def _run(work):
    with _lock:
        connection = get_connection()
        try:
            return work(connection)
        except sqlite3.Error as error:
            raise DatabaseError(str(error)) from error


def _in_transaction(connection, action):
    connection.execute("BEGIN IMMEDIATE")
    try:
        result = action()
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")
    return result


def _now():
    return datetime.now().replace(microsecond=0)


def _to_text(value):
    return value.strftime(DATETIME_FORMAT)


def _to_datetime(text):
    if not text:
        return None
    try:
        return datetime.strptime(text, DATETIME_FORMAT)
    except (TypeError, ValueError):
        return None


def _b64encode(data):
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, HASH_ITERATIONS)
    return f"{HASH_PREFIX}${HASH_ITERATIONS}${_b64encode(salt)}${_b64encode(digest)}"


def is_hashed(stored):
    return isinstance(stored, str) and stored.startswith(HASH_PREFIX + "$")


def verify_password(password, stored):
    if not isinstance(stored, str):
        return False
    if is_hashed(stored):
        parts = stored.split("$")
        if len(parts) != 4:
            return False
        try:
            iterations = int(parts[1])
            salt = _b64decode(parts[2])
            expected = _b64decode(parts[3])
        except ValueError:
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(digest, expected)
    return hmac.compare_digest(password.encode("utf-8"), stored.encode("utf-8"))


def initialise_database(datasets_dir):
    def work(connection):
        connection.executescript(SCHEMA)

        def seed():
            if connection.execute("SELECT COUNT(*) FROM admin").fetchone()[0] == 0:
                for admin_id, password in DEFAULT_ADMINS:
                    connection.execute(
                        "INSERT INTO admin (admin_id, password, created_date) VALUES (?, ?, ?)",
                        (admin_id, hash_password(password), _to_text(_now())),
                    )
            creator = connection.execute("SELECT admin_id FROM admin ORDER BY created_date, admin_id LIMIT 1").fetchone()[0]
            if not os.path.isdir(datasets_dir):
                return
            existing = {row[0].lower() for row in connection.execute("SELECT user_id FROM users")}
            for name in sorted(os.listdir(datasets_dir)):
                if not os.path.isdir(os.path.join(datasets_dir, name)):
                    continue
                if name.lower() in existing or name.lower() == "unknown" or len(name) >= 100:
                    continue
                connection.execute(
                    "INSERT INTO users (user_id, department, face_image_path, created_by) VALUES (?, ?, ?, ?)",
                    (name, None, os.path.join(os.path.basename(datasets_dir), name), creator),
                )
                existing.add(name.lower())

        _in_transaction(connection, seed)

    try:
        _run(work)
    except DatabaseError as error:
        raise DatabaseError(f"Database initialisation failed: {error}") from error


def admin_exists(admin_id):
    def work(connection):
        return connection.execute("SELECT 1 FROM admin WHERE admin_id = ?", (admin_id,)).fetchone() is not None

    return _run(work)


def verify_admin(admin_id, password):
    def work(connection):
        row = connection.execute("SELECT password FROM admin WHERE admin_id = ?", (admin_id,)).fetchone()
        if row is None or not verify_password(password, row[0]):
            return False
        if not is_hashed(row[0]):
            connection.execute("UPDATE admin SET password = ? WHERE admin_id = ?", (hash_password(password), admin_id))
        return True

    return _run(work)


def insert_admin(admin_id, password):
    def work(connection):
        connection.execute(
            "INSERT INTO admin (admin_id, password, created_date) VALUES (?, ?, ?)",
            (admin_id, hash_password(password), _to_text(_now())),
        )

    _run(work)


def user_exists(user_id):
    def work(connection):
        return connection.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone() is not None

    return _run(work)


def insert_user(user_id, department, image_path, created_by):
    def work(connection):
        connection.execute(
            "INSERT INTO users (user_id, department, face_image_path, created_by) VALUES (?, ?, ?, ?)",
            (user_id, department, image_path, created_by),
        )

    _run(work)


def _user_from_row(row):
    return {"user_id": row[0], "department": row[1] or "", "face_image_path": row[2], "created_by": row[3]}


def get_users():
    def work(connection):
        rows = connection.execute(
            "SELECT user_id, department, face_image_path, created_by FROM users ORDER BY user_id COLLATE NOCASE"
        ).fetchall()
        return [_user_from_row(row) for row in rows]

    return _run(work)


def get_user(user_id):
    def work(connection):
        row = connection.execute(
            "SELECT user_id, department, face_image_path, created_by FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return _user_from_row(row) if row else None

    return _run(work)


def update_user_department(user_id, department):
    def work(connection):
        cursor = connection.execute("UPDATE users SET department = ? WHERE user_id = ?", (department, user_id))
        return cursor.rowcount > 0

    return _run(work)


def delete_user(user_id):
    def work(connection):
        def remove():
            connection.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
            return connection.execute("DELETE FROM users WHERE user_id = ?", (user_id,)).rowcount > 0

        return _in_transaction(connection, remove)

    return _run(work)


def has_open_attendance(user_id):
    def work(connection):
        row = connection.execute(
            "SELECT 1 FROM attendance WHERE user_id = ? AND exit_date_time IS NULL",
            (user_id,),
        ).fetchone()
        return row is not None

    return _run(work)


def insert_attendance(user_id):
    entry_time = _now()

    def work(connection):
        connection.execute(
            "INSERT INTO attendance (entry_date_time, exit_date_time, user_id) VALUES (?, NULL, ?)",
            (_to_text(entry_time), user_id),
        )
        return entry_time

    return _run(work)


def close_attendance(user_id):
    exit_time = _now()

    def work(connection):
        cursor = connection.execute(
            "UPDATE attendance SET exit_date_time = ? WHERE user_id = ? AND exit_date_time IS NULL",
            (_to_text(exit_time), user_id),
        )
        return exit_time if cursor.rowcount > 0 else None

    return _run(work)


def _escape_like(text):
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def get_attendance_records(user_filter=None, date_from=None, date_to=None):
    sql = (
        "SELECT a.attendance_id, a.user_id, u.department, a.entry_date_time, a.exit_date_time "
        "FROM attendance a LEFT JOIN users u ON u.user_id = a.user_id WHERE 1 = 1"
    )
    params = []
    if user_filter:
        sql += " AND a.user_id LIKE ? ESCAPE '\\'"
        params.append(f"%{_escape_like(user_filter)}%")
    if date_from:
        sql += " AND a.entry_date_time >= ?"
        params.append(_to_text(datetime.combine(date_from, dt_time.min)))
    if date_to:
        sql += " AND a.entry_date_time < ?"
        params.append(_to_text(datetime.combine(date_to + timedelta(days=1), dt_time.min)))
    sql += " ORDER BY a.entry_date_time DESC, a.attendance_id DESC"

    def work(connection):
        records = []
        for row in connection.execute(sql, params).fetchall():
            records.append(
                {
                    "attendance_id": row[0],
                    "user_id": row[1],
                    "department": row[2] or "",
                    "entry": _to_datetime(row[3]),
                    "exit": _to_datetime(row[4]),
                }
            )
        return records

    return _run(work)
