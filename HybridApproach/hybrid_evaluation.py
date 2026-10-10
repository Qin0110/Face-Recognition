import os

import numpy as np

from face_utils import list_images, list_subdirectories, read_image
from HybridApproach.hybrid import calc_metrics, encode_face, load_dataset_faces_from_json, match_encoding
from preprocess import PreProcess


class Result:
    def __init__(self):
        self.train_names, self.train_encodings = load_dataset_faces_from_json()

    def evaluation_dataset_face(self, test_dir=PreProcess.PREDICTION_DIR, show_plots=True):
        if not self.train_names:
            raise RuntimeError("No known faces found. Please run 'Train Model' first.")

        tp = tn = fp = fn = 0
        known = set(self.train_names)
        for person_name in list_subdirectories(test_dir):
            for image_path in list_images(os.path.join(test_dir, person_name)):
                encoding = encode_face(read_image(image_path), allow_whole_image=False)
                predicted = None
                if encoding is not None:
                    predicted = match_encoding(encoding, self.train_names, self.train_encodings)[0]
                if person_name in known:
                    if predicted == person_name:
                        tp += 1
                    else:
                        fn += 1
                elif predicted is not None:
                    fp += 1
                else:
                    tn += 1

        precision, recall, f1, accuracy = calc_metrics(tp, tn, fp, fn)
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}")

        if show_plots:
            self._plot(tp, tn, fp, fn, precision, recall, f1, accuracy)
        return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

    @staticmethod
    def _plot(tp, tn, fp, fn, precision, recall, f1, accuracy):
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 4))
        plt.bar(["Accuracy", "Precision", "Recall", "F1 Score"], [accuracy, precision, recall, f1], color=["skyblue", "orange", "green", "red"])
        plt.ylim(0, 1)
        plt.title("Model Performance Metrics")
        plt.ylabel("Score")
        plt.show()

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            np.array([[tp, fn], [fp, tn]]),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Positive", "Predicted Negative"],
            yticklabels=["Actual Positive", "Actual Negative"],
        )
        plt.title("Confusion Matrix")
        plt.show()

    def identify_user(self, img):
        if not self.train_names:
            return None
        encoding = encode_face(img)
        if encoding is None:
            return None
        return match_encoding(encoding, self.train_names, self.train_encodings)[0]


if __name__ == "__main__":
    Result().evaluation_dataset_face()
