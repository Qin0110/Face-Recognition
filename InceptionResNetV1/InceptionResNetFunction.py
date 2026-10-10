import csv
import json
import os
import threading
import time

import cv2
import numpy as np

from face_utils import list_images, list_subdirectories, prepare_face, read_image
from preprocess import PreProcess

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(MODULE_DIR), "models")
os.environ["TORCH_HOME"] = MODELS_DIR
EMBEDDINGS_PATH = os.path.join(MODULE_DIR, "embeddings.json")
FINE_TUNED_MODEL_PATH = os.path.join(MODULE_DIR, "final_model.pth")
RESULTS_CSV_PATH = os.path.join(MODULE_DIR, "evaluation_results.csv")
CONFUSION_MATRIX_PATH = os.path.join(MODULE_DIR, "Confusion_Matrix.png")
CLASS_CONFUSION_MATRIX_PATH = os.path.join(MODULE_DIR, "Cosine Similarity Confusion Matrix with all classes.png")

EMBEDDING_SIZE = 512
IMAGE_SIZE = 160
BATCH_SIZE = 32
TOP_K = 3
DEFAULT_THRESHOLD = 0.6
MIN_THRESHOLD = 0.4
MAX_THRESHOLD = 0.85
UNKNOWN_LABEL = "Unknown"
NO_FACE_LABEL = "No Face Detected"
PRETRAINED_SIGNATURE = "vggface2"

_model_lock = threading.Lock()
_gallery_lock = threading.Lock()
_model = None
_device = None
_model_signature = None
_gallery = None
_gallery_mtime = None


class Gallery:
    def __init__(self, labels, embeddings, threshold, signature):
        self.labels = [str(label) for label in labels]
        self.embeddings = np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_SIZE)
        self.threshold = float(threshold)
        self.signature = signature
        self.classes = sorted(set(self.labels))
        self.class_lookup = {name: position for position, name in enumerate(self.classes)}
        label_array = np.array(self.labels, dtype=object)
        self.class_indices = [np.flatnonzero(label_array == name) for name in self.classes]

    def class_scores(self, embedding):
        similarities = self.embeddings @ embedding
        scores = np.empty(len(self.classes), dtype=np.float32)
        for position, indices in enumerate(self.class_indices):
            values = similarities[indices]
            k = min(TOP_K, values.size)
            top_values = np.partition(values, values.size - k)[values.size - k:]
            scores[position] = float(np.mean(top_values))
        return scores

    def match(self, embedding):
        if not self.classes:
            return None, 0.0
        scores = self.class_scores(embedding)
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score >= self.threshold:
            return self.classes[best], score
        return None, score

    def to_json(self):
        return {
            "model": self.signature,
            "threshold": round(self.threshold, 4),
            "labels": self.labels,
            "embeddings": np.round(self.embeddings.astype(np.float64), 6).tolist(),
        }


def _build_pretrained_model():
    from facenet_pytorch import InceptionResnetV1

    return InceptionResnetV1(pretrained="vggface2")


def _read_fine_tuned_state(torch):
    try:
        checkpoint = torch.load(FINE_TUNED_MODEL_PATH, map_location="cpu", weights_only=True)
    except Exception:
        try:
            checkpoint = torch.load(FINE_TUNED_MODEL_PATH, map_location="cpu", weights_only=False)
        except Exception:
            return None
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, torch.nn.Module):
                checkpoint = value.state_dict()
                break
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        return None
    state = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        name = key[len("module."):] if key.startswith("module.") else key
        if name.startswith("logits."):
            continue
        state[name] = value
    return state or None


def load_model():
    global _model, _device, _model_signature
    with _model_lock:
        if _model is not None:
            return _model
        try:
            import torch

            model = _build_pretrained_model()
        except Exception as error:
            raise RuntimeError(
                "Unable to load the InceptionResNetV1 model. The pretrained weights are downloaded on first use, "
                f"so check the internet connection and the installed packages. Details: {error}"
            ) from error
        signature = PRETRAINED_SIGNATURE
        if os.path.isfile(FINE_TUNED_MODEL_PATH):
            state = _read_fine_tuned_state(torch)
            if state is not None:
                try:
                    result = model.load_state_dict(state, strict=False)
                    missing = [key for key in result.missing_keys if not key.startswith("logits.")]
                    if missing:
                        model = _build_pretrained_model()
                    else:
                        signature = f"fine-tuned:{int(os.path.getmtime(FINE_TUNED_MODEL_PATH))}"
                except Exception:
                    model = _build_pretrained_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)
        _model, _device, _model_signature = model, device, signature
        return _model


def is_model_loaded():
    return _model is not None


def _prepare_batch(faces):
    import torch

    arrays = []
    for face in faces:
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        interpolation = cv2.INTER_AREA if min(height, width) > IMAGE_SIZE else cv2.INTER_LINEAR
        arrays.append(cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interpolation))
    batch = (np.stack(arrays).astype(np.float32) - 127.5) / 128.0
    return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()


def compute_embeddings(faces):
    if not faces:
        return np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)
    import torch

    model = load_model()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(faces), BATCH_SIZE):
            batch = _prepare_batch(faces[start:start + BATCH_SIZE]).to(_device)
            outputs.append(model(batch).detach().cpu().numpy())
    embeddings = np.concatenate(outputs, axis=0).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-10)


def _load_gallery_file():
    try:
        with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    labels = data.get("labels")
    embeddings = data.get("embeddings")
    signature = data.get("model")
    if not isinstance(labels, list) or not isinstance(embeddings, list) or not signature:
        return None
    if not labels or len(labels) != len(embeddings):
        return None
    try:
        array = np.asarray(embeddings, dtype=np.float32)
        threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[1] != EMBEDDING_SIZE:
        return None
    return Gallery(labels, array, threshold, signature)


def _save_gallery(gallery):
    global _gallery, _gallery_mtime
    temp_path = EMBEDDINGS_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(gallery.to_json(), file)
    os.replace(temp_path, EMBEDDINGS_PATH)
    with _gallery_lock:
        _gallery = gallery if gallery.labels else None
        _gallery_mtime = os.path.getmtime(EMBEDDINGS_PATH)


def get_gallery():
    global _gallery, _gallery_mtime
    try:
        mtime = os.path.getmtime(EMBEDDINGS_PATH)
    except OSError:
        with _gallery_lock:
            _gallery = None
            _gallery_mtime = None
        return None
    with _gallery_lock:
        if mtime != _gallery_mtime:
            _gallery = _load_gallery_file()
            _gallery_mtime = mtime
        return _gallery


def _expected_signature():
    if _model_signature is not None:
        return _model_signature
    if os.path.isfile(FINE_TUNED_MODEL_PATH):
        return None
    return PRETRAINED_SIGNATURE


def check_data_exists():
    gallery = get_gallery()
    if gallery is None:
        return False
    expected = _expected_signature()
    return expected is None or gallery.signature == expected


def identify_with_score(face):
    if face is None or face.size == 0:
        return None, 0.0
    gallery = get_gallery()
    if gallery is None:
        raise RuntimeError("No trained data found. Please run 'Train Model' first.")
    load_model()
    if gallery.signature != _model_signature:
        raise RuntimeError("The trained data was created with a different model. Please run 'Train Model' again.")
    embedding = compute_embeddings([face])[0]
    return gallery.match(embedding)


def identify_user(face):
    return identify_with_score(face)[0]


def remove_user(name):
    gallery = get_gallery()
    if gallery is None or name not in gallery.class_lookup:
        return False
    keep = np.array([index for index, label in enumerate(gallery.labels) if label != name], dtype=np.intp)
    remaining = Gallery([gallery.labels[index] for index in keep], gallery.embeddings[keep], gallery.threshold, gallery.signature)
    _save_gallery(remaining)
    return True


def _embed_directory(directory, report, stage):
    labels = []
    embeddings = []
    people = PreProcess.list_people(directory)
    for index, name in enumerate(people, start=1):
        report(f"Extracting {stage} embeddings ({index}/{len(people)}): {name}")
        faces = []
        for path in list_images(os.path.join(directory, name)):
            image = read_image(path)
            if image is not None and image.size > 0:
                faces.append(image)
        if not faces:
            continue
        vectors = compute_embeddings(faces)
        labels.extend([name] * len(vectors))
        embeddings.append(vectors)
    if not embeddings:
        return [], np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)
    return labels, np.concatenate(embeddings, axis=0)


def _calibrate_threshold(gallery, labels, embeddings):
    if len(gallery.classes) < 2 or not labels:
        return DEFAULT_THRESHOLD
    genuine = []
    impostor = []
    for label, embedding in zip(labels, embeddings):
        position = gallery.class_lookup.get(label)
        if position is None:
            continue
        scores = gallery.class_scores(embedding)
        genuine.append(scores[position])
        impostor.append(np.max(np.delete(scores, position)))
    if not genuine:
        return DEFAULT_THRESHOLD
    genuine = np.asarray(genuine)
    impostor = np.asarray(impostor)
    candidates = np.round(np.arange(MIN_THRESHOLD, MAX_THRESHOLD + 0.001, 0.01), 2)
    balanced = np.array([(np.mean(genuine >= value) + np.mean(impostor < value)) / 2.0 for value in candidates])
    best = candidates[balanced >= balanced.max() - 1e-9]
    return float(np.median(best))


def _compute_metrics(gallery, true_labels, predicted_labels):
    tp = tn = fp = fn = 0
    for actual, predicted in zip(true_labels, predicted_labels):
        if actual in gallery.class_lookup:
            if predicted == actual:
                tp += 1
            else:
                fn += 1
        elif predicted in gallery.class_lookup:
            fp += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": (tp + tn) / total if total else 0.0,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _write_results(metrics):
    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def _draw_matrix(matrix, row_labels, column_labels, title, path, figsize, xlabel=None, ylabel=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    image = axes.imshow(matrix, cmap="Blues", aspect="auto")
    figure.colorbar(image, ax=axes)
    axes.set_xticks(range(len(column_labels)))
    axes.set_xticklabels(column_labels, rotation=90 if len(column_labels) > 2 else 0)
    axes.set_yticks(range(len(row_labels)))
    axes.set_yticklabels(row_labels)
    limit = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0
    font_size = 8 if len(column_labels) > 10 else 12
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            color = "white" if value > limit else "black"
            axes.text(column, row, str(value), ha="center", va="center", color=color, fontsize=font_size)
    axes.set_title(title)
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylabel:
        axes.set_ylabel(ylabel)
    figure.tight_layout()
    figure.savefig(path)


def _save_confusion_matrices(metrics, true_labels, predicted_labels):
    try:
        binary = np.array([[metrics["TP"], metrics["FN"]], [metrics["FP"], metrics["TN"]]])
        _draw_matrix(
            binary,
            ["Actual Positive", "Actual Negative"],
            ["Predicted Positive", "Predicted Negative"],
            "Confusion Matrix",
            CONFUSION_MATRIX_PATH,
            (10, 5),
        )
        names = sorted((set(true_labels) | set(predicted_labels)) - {NO_FACE_LABEL})
        names.append(NO_FACE_LABEL)
        position = {name: index for index, name in enumerate(names)}
        matrix = np.zeros((len(names), len(names)), dtype=int)
        for actual, predicted in zip(true_labels, predicted_labels):
            matrix[position[actual], position[predicted]] += 1
        size = max(8.0, len(names) * 0.6)
        _draw_matrix(
            matrix,
            names,
            names,
            "Detailed Confusion Matrix for every class",
            CLASS_CONFUSION_MATRIX_PATH,
            (size, size),
            "Predicted Labels",
            "True Labels",
        )
    except Exception as error:
        return f"Confusion matrix images could not be saved: {error}"
    return None


def _evaluate(gallery, report):
    true_labels = []
    predicted_labels = []
    folders = list_subdirectories(PreProcess.PREDICTION_DIR)
    for index, name in enumerate(folders, start=1):
        report(f"Evaluating ({index}/{len(folders)}): {name}")
        faces = []
        for path in list_images(os.path.join(PreProcess.PREDICTION_DIR, name)):
            face = prepare_face(read_image(path))
            if face is None:
                true_labels.append(name)
                predicted_labels.append(NO_FACE_LABEL)
            else:
                faces.append(face)
        for embedding in compute_embeddings(faces):
            predicted, _ = gallery.match(embedding)
            true_labels.append(name)
            predicted_labels.append(predicted or UNKNOWN_LABEL)
    if not true_labels:
        return None, None
    metrics = _compute_metrics(gallery, true_labels, predicted_labels)
    _write_results(metrics)
    warning = _save_confusion_matrices(metrics, true_labels, predicted_labels)
    return metrics, warning


def train_model(progress=None):
    report = progress or (lambda message: None)
    started = time.monotonic()
    report("Loading InceptionResNetV1 model...")
    load_model()
    summary = PreProcess(progress=report).run()
    labels, embeddings = _embed_directory(PreProcess.TRAIN_DIR, report, "training")
    if not labels:
        raise RuntimeError(
            f"No faces are available for training. Each user needs at least {PreProcess.MIN_IMAGES} face images."
        )
    gallery = Gallery(labels, embeddings, DEFAULT_THRESHOLD, _model_signature)
    test_labels, test_embeddings = _embed_directory(PreProcess.TEST_DIR, report, "validation")
    report("Calibrating similarity threshold...")
    gallery.threshold = _calibrate_threshold(gallery, test_labels, test_embeddings)
    report("Saving trained data...")
    _save_gallery(gallery)
    metrics, warning = _evaluate(gallery, report)

    lines = [
        f"Training completed in {time.monotonic() - started:.0f} seconds.",
        f"Users trained: {len(gallery.classes)} ({len(labels)} face images).",
        f"Similarity threshold: {gallery.threshold:.2f}",
    ]
    if metrics:
        lines.append(
            f"Accuracy: {metrics['Accuracy']:.1%}   Precision: {metrics['Precision']:.1%}   "
            f"Recall: {metrics['Recall']:.1%}   F1 Score: {metrics['F1 Score']:.1%}"
        )
    if summary["skipped"]:
        skipped = ", ".join(f"{name} ({count})" for name, count in summary["skipped"])
        lines.append(f"Skipped users with fewer than {PreProcess.MIN_IMAGES} images: {skipped}")
    if summary["dropped"]:
        lines.append(f"Images without a detectable face ignored: {summary['dropped']}")
    if warning:
        lines.append(warning)
    return "\n".join(lines)


if __name__ == "__main__":
    print(train_model(print))
