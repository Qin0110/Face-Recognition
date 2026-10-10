import json
import os
import threading
import time

import cv2
import numpy as np
from numpy import dot
from numpy.linalg import norm

from face_utils import list_images, read_image
from preprocess import PreProcess

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FACE_PATH = os.path.join(MODULE_DIR, "dataset_face.json")
TRAIN_DIR = PreProcess.TRAIN_DIR
COSINE_THRESHOLD = 0.7
EUCLIDEAN_THRESHOLD = 0.4

_cache_lock = threading.Lock()
_cache = {"mtime": None, "names": [], "encodings": []}


def cosine_similarity(vec1, vec2):
    denominator = norm(vec1) * norm(vec2)
    if denominator == 0:
        return 0.0
    return float(dot(vec1, vec2) / denominator)


def euclidean_distance(vec1, vec2):
    return float(norm(np.asarray(vec1) - np.asarray(vec2)))


def adjust_gamma_auto(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 80:
        gamma = 1.8
    elif mean_brightness < 120:
        gamma = 1.4
    elif mean_brightness > 200:
        gamma = 0.8
    else:
        gamma = 1.0
    inverse_gamma = 1.0 / gamma
    table = (((np.arange(256) / 255.0) ** inverse_gamma) * 255).astype("uint8")
    return cv2.LUT(image, table), gamma


def calc_metrics(tp, tn, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0
    return precision, recall, f1, accuracy


def load_model():
    try:
        import face_recognition
    except (ImportError, SystemExit) as error:
        raise RuntimeError("face_recognition is not installed correctly. Please run setup.bat again.") from error
    return face_recognition


def encode_face(image, allow_whole_image=True):
    face_recognition = load_model()
    if image is None or image.size == 0:
        return None
    adjusted, _ = adjust_gamma_auto(image)
    rgb = np.ascontiguousarray(cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB))
    locations = face_recognition.face_locations(rgb)
    if not locations:
        if not allow_whole_image:
            return None
        height, width = rgb.shape[:2]
        locations = [(0, width, height, 0)]
    encodings = face_recognition.face_encodings(rgb, locations)
    return encodings[0] if encodings else None


def match_encoding(encoding, names, encodings):
    best_name = None
    best_similarity = 0.0
    for name, known in zip(names, encodings):
        cos_sim = cosine_similarity(encoding, known)
        euc_dist = euclidean_distance(encoding, known)
        if cos_sim > COSINE_THRESHOLD and euc_dist < EUCLIDEAN_THRESHOLD and cos_sim > best_similarity:
            best_similarity = cos_sim
            best_name = name
    return best_name, best_similarity


def train_dataset_face(progress=None, run_preprocess=True):
    report = progress or print
    summary = PreProcess(progress=report).run() if run_preprocess else {"included": [], "skipped": [], "dropped": 0}
    train_names = []
    train_encodings = []
    image_count = 0
    people = PreProcess.list_people(TRAIN_DIR)
    for index, person_name in enumerate(people, start=1):
        report(f"Encoding faces ({index}/{len(people)}): {person_name}")
        person_encodings = []
        for image_path in list_images(os.path.join(TRAIN_DIR, person_name)):
            try:
                encoding = encode_face(read_image(image_path))
            except Exception:
                encoding = None
            if encoding is not None:
                person_encodings.append(encoding)
        if person_encodings:
            train_names.append(person_name)
            train_encodings.append(np.mean(person_encodings, axis=0))
            image_count += len(person_encodings)
    save_dataset_faces_to_json(train_names, train_encodings)
    return train_names, image_count, summary


def save_dataset_faces_to_json(train_names, train_encodings, filepath=DATASET_FACE_PATH):
    data = [
        {"name": name, "encoding": np.asarray(encoding, dtype=float).tolist()}
        for name, encoding in zip(train_names, train_encodings)
    ]
    temp_path = filepath + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file)
    os.replace(temp_path, filepath)
    return len(data)


def load_dataset_faces_from_json(filepath=DATASET_FACE_PATH):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return [], []
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return [], []
    if not isinstance(data, list):
        return [], []
    known_names = []
    known_encodings = []
    for item in data:
        if not isinstance(item, dict) or "name" not in item or "encoding" not in item:
            continue
        known_names.append(item["name"])
        known_encodings.append(np.array(item["encoding"], dtype=float))
    return known_names, known_encodings


def get_known_faces():
    try:
        mtime = os.path.getmtime(DATASET_FACE_PATH)
    except OSError:
        return [], []
    with _cache_lock:
        if _cache["mtime"] != mtime:
            names, encodings = load_dataset_faces_from_json()
            _cache.update(mtime=mtime, names=names, encodings=encodings)
        return _cache["names"], _cache["encodings"]


def check_data_exists():
    names, _ = get_known_faces()
    return bool(names)


def identify_user(face):
    names, encodings = get_known_faces()
    if not names:
        raise RuntimeError("No trained data found. Please run 'Train Model' first.")
    encoding = encode_face(face)
    if encoding is None:
        return None
    return match_encoding(encoding, names, encodings)[0]


def remove_user(name):
    names, encodings = load_dataset_faces_from_json()
    if name not in names:
        return False
    kept = [(known_name, encoding) for known_name, encoding in zip(names, encodings) if known_name != name]
    save_dataset_faces_to_json([item[0] for item in kept], [item[1] for item in kept])
    return True


def train_model(progress=None):
    report = progress or (lambda message: None)
    started = time.monotonic()
    report("Loading face_recognition model...")
    load_model()
    names, image_count, summary = train_dataset_face(report)
    if not names:
        raise RuntimeError(
            f"No faces are available for training. Each user needs at least {PreProcess.MIN_IMAGES} face images."
        )
    lines = [
        f"Training completed in {time.monotonic() - started:.0f} seconds.",
        f"Saved embeddings for {len(names)} users ({image_count} face images).",
    ]
    if summary["skipped"]:
        skipped = ", ".join(f"{name} ({count})" for name, count in summary["skipped"])
        lines.append(f"Skipped users with fewer than {PreProcess.MIN_IMAGES} images: {skipped}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(train_model(print))
