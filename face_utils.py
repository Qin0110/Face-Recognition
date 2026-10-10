import os
import re
import threading

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
DETECTION_MAX_SIDE = 800
FULL_FACE_RATIO = 0.6

_local = threading.local()


def get_face_cascade():
    cascade = getattr(_local, "cascade", None)
    if cascade is None:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError("Unable to load the Haar cascade face detector.")
        _local.cascade = cascade
    return cascade


def natural_key(path):
    name = os.path.basename(path).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def list_images(directory):
    if not os.path.isdir(directory):
        return []
    images = []
    for file_name in os.listdir(directory):
        path = os.path.join(directory, file_name)
        if file_name.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(path):
            images.append(path)
    return sorted(images, key=natural_key)


def list_subdirectories(directory, exclude=()):
    if not os.path.isdir(directory):
        return []
    excluded = {name.lower() for name in exclude}
    names = []
    for name in os.listdir(directory):
        if name.lower() in excluded:
            continue
        if os.path.isdir(os.path.join(directory, name)):
            names.append(name)
    return sorted(names, key=str.lower)


def read_image(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image):
    extension = os.path.splitext(path)[1] or ".jpg"
    success, buffer = cv2.imencode(extension, image)
    if not success:
        return False
    try:
        buffer.tofile(path)
    except OSError:
        return False
    return True


def to_gray(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_faces(image, max_side=DETECTION_MAX_SIDE):
    if image is None or image.size == 0:
        return []
    gray = to_gray(image)
    height, width = gray.shape[:2]
    scale = 1.0
    if max(height, width) > max_side:
        scale = max_side / float(max(height, width))
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    detections = get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    faces = []
    for (x, y, w, h) in detections:
        faces.append((int(x / scale), int(y / scale), int(w / scale), int(h / scale)))
    return faces


def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda face: face[2] * face[3])


def crop_face(image, face):
    x, y, w, h = face
    height, width = image.shape[:2]
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + w), min(height, y + h)
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def prepare_face(image, full_face_ratio=FULL_FACE_RATIO):
    if image is None or image.size == 0:
        return None
    face = largest_face(detect_faces(image))
    if face is None:
        return None
    height, width = image.shape[:2]
    if face[2] * face[3] >= full_face_ratio * width * height:
        return image
    return crop_face(image, face)


def blur_score(image):
    return float(cv2.Laplacian(to_gray(image), cv2.CV_64F).var())
