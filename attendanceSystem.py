import csv
import os
import re
import shutil
import threading
import time
from collections import Counter
from datetime import datetime

import cv2

import database
from face_utils import blur_score, crop_face, detect_faces, list_images, write_image
from preprocess import PreProcess

DATASETS_DIR = PreProcess.SOURCE_DIR
RECOGNITION_ENGINE = os.environ.get("FRAS_ENGINE", "inception").strip().lower()
UNKNOWN_LABEL = "Unknown"
MAX_ID_LENGTH = 100
MIN_PASSWORD_LENGTH = 5
CAPTURE_TARGET = PreProcess.MIN_IMAGES
CAPTURE_SIZE = 250
CAPTURE_INTERVAL_SECONDS = 0.15
ATTENDANCE_SAMPLES = 30
ATTENDANCE_MIN_SAMPLES = 10
ATTENDANCE_MIN_RATIO = 0.6
ATTENDANCE_TIMEOUT_SECONDS = 60
MIN_BLUR_SCORE = 30
MIN_FACE_RATIO = 0.02
MAX_FACE_RATIO = 0.8
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9 ._-]*[A-Za-z0-9_-])?$")
RESERVED_NAMES = (
    {"CON", "PRN", "AUX", "NUL", UNKNOWN_LABEL.upper()}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)

GREEN = (0, 200, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)

_model_state = {"loading": False, "error": None}


def get_engine():
    if RECOGNITION_ENGINE == "hybrid":
        from HybridApproach import hybrid

        return hybrid
    from InceptionResNetV1 import InceptionResNetFunction

    return InceptionResNetFunction


def preload_model():
    def worker():
        try:
            get_engine().load_model()
            _model_state["error"] = None
        except Exception as error:
            _model_state["error"] = str(error)
        finally:
            _model_state["loading"] = False

    _model_state["loading"] = True
    threading.Thread(target=worker, name="model-preload", daemon=True).start()


def model_error():
    return _model_state["error"]


def initialise_system():
    try:
        os.makedirs(DATASETS_DIR, exist_ok=True)
        database.initialise_database(DATASETS_DIR)
    except (database.DatabaseError, OSError) as error:
        return False, str(error)
    return True, None


def shutdown():
    database.close_connection()


def _user_directory(root, user_id):
    if not user_id:
        return None
    root_path = os.path.abspath(root)
    path = os.path.abspath(os.path.join(root_path, user_id))
    if os.path.dirname(path) != root_path:
        return None
    return path


def validate_admin_id(admin_id):
    if not admin_id:
        return "Admin ID cannot be empty."
    if len(admin_id) >= MAX_ID_LENGTH:
        return f"Admin ID must be less than {MAX_ID_LENGTH} characters."
    return None


def validate_password(password):
    if not password:
        return "Password cannot be empty."
    if any(character.isspace() for character in password):
        return "Password cannot contain spaces."
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if len(password) >= MAX_ID_LENGTH:
        return f"Password must be less than {MAX_ID_LENGTH} characters."
    return None


def validate_user_id(user_id):
    if not user_id:
        return "User ID cannot be empty."
    if len(user_id) >= MAX_ID_LENGTH:
        return f"User ID must be less than {MAX_ID_LENGTH} characters."
    if not USER_ID_PATTERN.match(user_id):
        return (
            "User ID must start with a letter or number and may only contain letters, numbers, "
            "spaces, '.', '_' and '-'. It cannot end with a space or '.'."
        )
    if user_id.upper() in RESERVED_NAMES:
        return f"'{user_id}' is a reserved name. Please choose another User ID."
    return None


def login(admin_id, password):
    if not admin_id or not password:
        return False, "Please enter both Admin ID and Password."
    if len(admin_id) >= MAX_ID_LENGTH or len(password) >= MAX_ID_LENGTH:
        return False, "Admin ID or Password is incorrect."
    try:
        if database.verify_admin(admin_id, password):
            return True, None
    except database.DatabaseError as error:
        return False, f"Database error: {error}"
    return False, "Admin ID or Password is incorrect."


def register_admin(admin_id, password, confirm_password):
    message = validate_admin_id(admin_id) or validate_password(password)
    if message:
        return False, message
    if password != confirm_password:
        return False, "Passwords do not match."
    try:
        if database.admin_exists(admin_id):
            return False, f"Admin '{admin_id}' already exists."
        database.insert_admin(admin_id, password)
    except database.DatabaseError as error:
        return False, f"Database error: {error}"
    return True, f"Admin '{admin_id}' added successfully!"


def add_new_user(user_id, department, created_by):
    message = validate_user_id(user_id)
    if message:
        return False, message
    if len(department) >= MAX_ID_LENGTH:
        return False, f"Department must be less than {MAX_ID_LENGTH} characters."
    user_dir = _user_directory(DATASETS_DIR, user_id)
    if user_dir is None:
        return False, "Invalid User ID."
    try:
        if database.user_exists(user_id):
            return False, f"User '{user_id}' already exists."
    except database.DatabaseError as error:
        return False, f"Database error: {error}"
    if os.path.exists(user_dir):
        return False, f"A face image folder already exists for '{os.path.basename(user_dir)}'."
    try:
        os.makedirs(user_dir)
    except OSError as error:
        return False, f"Unable to create the face image folder: {error}"
    try:
        database.insert_user(user_id, department or None, os.path.join(os.path.basename(DATASETS_DIR), user_id), created_by)
    except database.DatabaseError as error:
        shutil.rmtree(user_dir, ignore_errors=True)
        return False, f"Database error: {error}"
    return True, f"User '{user_id}' added successfully!"


def count_trainable_users():
    total = 0
    for name in PreProcess.list_people(DATASETS_DIR):
        if len(list_images(os.path.join(DATASETS_DIR, name))) >= PreProcess.MIN_IMAGES:
            total += 1
    return total


def count_user_images(user_id):
    user_dir = _user_directory(DATASETS_DIR, user_id)
    return len(list_images(user_dir)) if user_dir else 0


def get_profile_image(user_id):
    user_dir = _user_directory(DATASETS_DIR, user_id)
    if user_dir is None:
        return None
    images = list_images(user_dir)
    return images[0] if images else None


def list_users():
    try:
        users = database.get_users()
    except database.DatabaseError as error:
        return False, f"Database error: {error}", []
    for user in users:
        user["image_count"] = count_user_images(user["user_id"])
    return True, None, users


def get_user_info(user_id):
    try:
        user = database.get_user(user_id)
    except database.DatabaseError as error:
        return False, f"Database error: {error}", None
    if user is None:
        return False, f"User '{user_id}' does not exist.", None
    user["image_count"] = count_user_images(user_id)
    user["profile_image"] = get_profile_image(user_id)
    return True, None, user


def update_user_department(user_id, department):
    department = department.strip()
    if len(department) >= MAX_ID_LENGTH:
        return False, f"Department must be less than {MAX_ID_LENGTH} characters."
    try:
        updated = database.update_user_department(user_id, department or None)
    except database.DatabaseError as error:
        return False, f"Failed to update {user_id}'s data: {error}"
    if not updated:
        return False, f"User '{user_id}' does not exist."
    return True, f"{user_id}'s data has been updated."


def delete_user(user_id):
    try:
        deleted = database.delete_user(user_id)
    except database.DatabaseError as error:
        return False, f"Failed to delete user {user_id}: {error}"
    if not deleted:
        return False, f"Failed to delete user {user_id}."

    problems = []
    roots = (DATASETS_DIR, PreProcess.RESCALE_DIR, PreProcess.TRAIN_DIR, PreProcess.TEST_DIR, PreProcess.PREDICTION_DIR)
    for root in roots:
        if root == PreProcess.PREDICTION_DIR and user_id.lower() == UNKNOWN_LABEL.lower():
            continue
        path = _user_directory(root, user_id)
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            if os.path.exists(path):
                problems.append(path)
    try:
        get_engine().remove_user(user_id)
    except Exception as error:
        problems.append(f"trained data ({error})")

    message = f"User {user_id} has been deleted."
    if problems:
        message += "\nSome files could not be removed:\n" + "\n".join(problems)
    return True, message


def prepare_recapture(user_id):
    user_dir = _user_directory(DATASETS_DIR, user_id)
    if user_dir is None:
        return False, "Invalid User ID."
    try:
        os.makedirs(user_dir, exist_ok=True)
        for path in list_images(user_dir):
            os.remove(path)
    except OSError as error:
        return False, f"Unable to remove the existing face images: {error}"
    return True, None


def record_attendance(user_id, status):
    try:
        if not database.user_exists(user_id):
            return False, f"User '{user_id}' does not exist. Please register first."
        if status == "Enter":
            if database.has_open_attendance(user_id):
                return False, f"{user_id} must check out before checking in again."
            entry_time = database.insert_attendance(user_id)
            return True, f"Entry recorded for {user_id} at {entry_time.strftime(DATETIME_FORMAT)}."
        if status == "Exit":
            exit_time = database.close_attendance(user_id)
            if exit_time is None:
                return False, f"{user_id} must check in before checking out."
            return True, f"Exit recorded for {user_id} at {exit_time.strftime(DATETIME_FORMAT)}."
    except database.DatabaseError as error:
        return False, f"Database error: {error}"
    return False, "Please select Enter or Exit."


def resolve_identity(votes, samples):
    if samples < ATTENDANCE_MIN_SAMPLES:
        return None, "No face could be verified clearly. Please face the camera and try again."
    known = [(name, count) for name, count in votes.items() if name != UNKNOWN_LABEL]
    if not known:
        return None, "Face detected, but not found in the trained data. Please register the user or run 'Train Model'."
    name, count = max(known, key=lambda item: item[1])
    if count / float(samples) < ATTENDANCE_MIN_RATIO:
        return None, "The face could not be recognised confidently. Please try again."
    return name, None


def format_datetime(value):
    return value.strftime(DATETIME_FORMAT) if value else ""


def format_duration(delta):
    seconds = max(0, int(delta.total_seconds()))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def parse_date(text, label):
    text = (text or "").strip()
    if not text:
        return None, None
    try:
        return datetime.strptime(text, DATE_FORMAT).date(), None
    except ValueError:
        return None, f"{label} date must be in YYYY-MM-DD format."


def get_attendance_records(user_filter, date_from_text, date_to_text):
    date_from, error = parse_date(date_from_text, "From")
    if error:
        return False, error, []
    date_to, error = parse_date(date_to_text, "To")
    if error:
        return False, error, []
    if date_from and date_to and date_from > date_to:
        return False, "The From date cannot be later than the To date.", []
    try:
        records = database.get_attendance_records((user_filter or "").strip() or None, date_from, date_to)
    except database.DatabaseError as error:
        return False, f"Database error: {error}", []
    for record in records:
        record["duration"] = format_duration(record["exit"] - record["entry"]) if record["exit"] and record["entry"] else ""
        record["status"] = "Completed" if record["exit"] else "Checked In"
    return True, None, records


def export_attendance_csv(records, path):
    try:
        with open(path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["Attendance ID", "User ID", "Department", "Entry", "Exit", "Duration", "Status"])
            for record in records:
                writer.writerow(
                    [
                        record["attendance_id"],
                        record["user_id"],
                        record["department"],
                        format_datetime(record["entry"]),
                        format_datetime(record["exit"]),
                        record["duration"],
                        record["status"],
                    ]
                )
    except OSError as error:
        return False, f"Unable to export the records: {error}"
    return True, f"Exported {len(records)} records to:\n{path}"


def check_data_exists():
    try:
        return bool(get_engine().check_data_exists())
    except Exception:
        return False


def train_model(progress=None):
    try:
        return True, get_engine().train_model(progress)
    except Exception as error:
        return False, f"Training failed: {error}"


def open_camera(index=0):
    backends = (cv2.CAP_DSHOW, cv2.CAP_ANY) if os.name == "nt" else (cv2.CAP_ANY,)
    for backend in backends:
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            return camera
        camera.release()
    return None


def close_camera(camera):
    if camera is not None:
        camera.release()


def verify_face(frame, faces):
    if not faces:
        return False, "No face detected"
    if len(faces) > 1:
        return False, "Only one face is allowed"
    if blur_score(frame) < MIN_BLUR_SCORE:
        return False, "Image is too blurry"
    x, y, w, h = faces[0]
    ratio = (w * h) / float(frame.shape[0] * frame.shape[1])
    if ratio < MIN_FACE_RATIO:
        return False, "Please move closer"
    if ratio > MAX_FACE_RATIO:
        return False, "Please move further away"
    return True, None


def _draw_box(image, face, color, label=None):
    x, y, w, h = face
    cv2.rectangle(image, (x - 2, y - 2), (x + w + 2, y + h + 2), color, 2)
    if label:
        cv2.putText(image, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _draw_banner(image, text, color):
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


class CameraSession:
    PREVIEW = "preview"
    CAPTURE = "capture"
    ATTENDANCE = "attendance"
    MAX_READ_FAILURES = 50
    MAX_WRITE_FAILURES = 10

    def __init__(self, mode, username=None, status=None, camera_index=0):
        if mode not in (self.PREVIEW, self.CAPTURE, self.ATTENDANCE):
            raise ValueError(f"Unknown camera mode: {mode}")
        self.mode = mode
        self.username = username
        self.status = status
        self.camera_index = camera_index
        self.captured = 0
        self.samples = 0
        self.votes = Counter()
        self.error = None
        self.cancelled = False
        self.timed_out = False
        self.message = "Opening camera..."
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._frame = None
        self._thread = None
        self._user_dir = None
        self._next_index = 0
        self._last_capture = 0.0
        self._write_failures = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.mode}", daemon=True)
        self._thread.start()

    def stop(self):
        if self.is_running():
            self.cancelled = True
        self._stop_event.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def take_frame(self):
        with self._frame_lock:
            frame, self._frame = self._frame, None
        return frame

    def _run(self):
        camera = None
        try:
            engine = None
            if self.mode == self.CAPTURE:
                self._prepare_capture()
            else:
                engine = get_engine()
            camera = open_camera(self.camera_index)
            if camera is None:
                self.error = "Could not open the camera. Please check that a webcam is connected and not in use by another application."
                return
            self.message = self._idle_message()
            started = time.monotonic()
            failures = 0
            while not self._stop_event.is_set():
                success, frame = camera.read()
                if not success or frame is None:
                    failures += 1
                    if failures >= self.MAX_READ_FAILURES:
                        self.error = "Unable to read frames from the camera."
                        break
                    time.sleep(0.02)
                    continue
                failures = 0
                display = frame.copy()
                faces = detect_faces(frame)
                if self.mode == self.CAPTURE:
                    done = self._handle_capture(frame, display, faces)
                else:
                    done = self._handle_recognition(frame, display, faces, engine)
                with self._frame_lock:
                    self._frame = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                if done:
                    break
                if self.mode == self.ATTENDANCE and time.monotonic() - started >= ATTENDANCE_TIMEOUT_SECONDS:
                    self.timed_out = True
                    break
        except Exception as error:
            self.error = str(error) or error.__class__.__name__
        finally:
            close_camera(camera)

    def _idle_message(self):
        if self.mode == self.CAPTURE:
            return f"Capturing faces for {self.username}: {self.captured}/{CAPTURE_TARGET}"
        if self.mode == self.ATTENDANCE:
            return f"Taking attendance ({self.status}): please look at the camera"
        return "Camera opened"

    def _prepare_capture(self):
        user_dir = _user_directory(DATASETS_DIR, self.username or "")
        if user_dir is None:
            raise ValueError("Invalid User ID for face capture.")
        os.makedirs(user_dir, exist_ok=True)
        pattern = re.compile(rf"^{re.escape(self.username)}_(\d+)\.jpg$", re.IGNORECASE)
        indices = []
        for path in list_images(user_dir):
            match = pattern.match(os.path.basename(path))
            if match:
                indices.append(int(match.group(1)))
        self._user_dir = user_dir
        self._next_index = max(indices) + 1 if indices else 0

    def _handle_capture(self, frame, display, faces):
        valid, reason = verify_face(frame, faces)
        if not valid:
            for face in faces:
                _draw_box(display, face, ORANGE)
            progress = f"Captured: {self.captured}/{CAPTURE_TARGET}"
            _draw_banner(display, f"{progress} - {reason}", ORANGE)
            self.message = f"Capturing faces for {self.username}: {progress} ({reason})"
            return False

        face = faces[0]
        now = time.monotonic()
        if now - self._last_capture >= CAPTURE_INTERVAL_SECONDS:
            crop = crop_face(frame, face)
            path = os.path.join(self._user_dir, f"{self.username}_{self._next_index}.jpg")
            if crop is not None and write_image(path, cv2.resize(crop, (CAPTURE_SIZE, CAPTURE_SIZE))):
                self.captured += 1
                self._next_index += 1
                self._last_capture = now
                self._write_failures = 0
            else:
                self._write_failures += 1
                if self._write_failures >= self.MAX_WRITE_FAILURES:
                    raise OSError(f"Unable to save face images to {self._user_dir}.")

        progress = f"Captured: {self.captured}/{CAPTURE_TARGET}"
        _draw_box(display, face, GREEN, progress)
        _draw_banner(display, progress, GREEN)
        self.message = f"Capturing faces for {self.username}: {progress}"
        return self.captured >= CAPTURE_TARGET

    def _handle_recognition(self, frame, display, faces, engine):
        if not faces:
            _draw_banner(display, "No faces detected", RED)
            self.message = self._idle_message()
            return False
        _draw_banner(display, f"Faces detected: {len(faces)}", GREEN)
        valid, reason = verify_face(frame, faces)
        if not valid:
            for face in faces:
                _draw_box(display, face, ORANGE, reason)
            self.message = reason
            return False

        face = faces[0]
        try:
            name = engine.identify_user(crop_face(frame, face))
        except Exception as error:
            if self.mode == self.ATTENDANCE:
                raise
            _draw_box(display, face, RED, "Error")
            self.message = f"Recognition unavailable: {error}"
            return False

        if name:
            _draw_box(display, face, GREEN, name)
        else:
            _draw_box(display, face, RED, UNKNOWN_LABEL)

        if self.mode != self.ATTENDANCE:
            self.message = f"Recognised: {name}" if name else "Unknown face"
            return False

        self.samples += 1
        self.votes[name or UNKNOWN_LABEL] += 1
        self.message = f"Taking attendance ({self.status}): {self.samples}/{ATTENDANCE_SAMPLES}"
        return self.samples >= ATTENDANCE_SAMPLES


def complete_session(session):
    if session.error:
        return False, session.error
    if session.mode == CameraSession.CAPTURE:
        if session.captured >= CAPTURE_TARGET:
            return True, (
                f"Captured {session.captured} face images for {session.username}.\n"
                "Run 'Train Model' to update the recognition data."
            )
        return False, (
            f"Face capture for {session.username} stopped at {session.captured}/{CAPTURE_TARGET} images.\n"
            "Use 'Retake Photos' in User Information to capture the faces again."
        )
    if session.mode == CameraSession.ATTENDANCE:
        if session.cancelled:
            return None, None
        name, message = resolve_identity(session.votes, session.samples)
        if name is None:
            return False, message
        return record_attendance(name, session.status)
    return None, None
