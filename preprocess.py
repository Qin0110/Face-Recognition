import os
import shutil

from face_utils import list_images, list_subdirectories, prepare_face, read_image, write_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class PreProcess:
    SOURCE_DIR = os.path.join(BASE_DIR, "Datasets")
    RESCALE_DIR = os.path.join(BASE_DIR, "Rescale_Datasets")
    DATA_DIR = os.path.join(BASE_DIR, "Data")
    TRAIN_DIR = os.path.join(DATA_DIR, "Train")
    TEST_DIR = os.path.join(DATA_DIR, "Test")
    PREDICTION_DIR = os.path.join(BASE_DIR, "Prediction")
    UNKNOWN_DIR_NAME = "Unknown"
    MIN_IMAGES = 70
    PREDICTION_IMAGES = 10
    TEST_SIZE = 0.3
    RANDOM_STATE = 42

    def __init__(self, progress=None):
        self.progress = progress or print

    @classmethod
    def list_people(cls, directory):
        return list_subdirectories(directory, exclude=(cls.UNKNOWN_DIR_NAME,))

    def run(self):
        self.reset_train_data()
        included, skipped = self.rescale()
        self.split_dataset()
        dropped = self.crop_faces()
        return {"included": included, "skipped": skipped, "dropped": dropped}

    def reset_train_data(self):
        self.progress("Clearing previous training data...")
        for directory in (self.TRAIN_DIR, self.TEST_DIR, self.RESCALE_DIR):
            shutil.rmtree(directory, ignore_errors=True)
        for name in self.list_people(self.PREDICTION_DIR):
            shutil.rmtree(os.path.join(self.PREDICTION_DIR, name), ignore_errors=True)

    def rescale(self):
        os.makedirs(self.RESCALE_DIR, exist_ok=True)
        os.makedirs(self.PREDICTION_DIR, exist_ok=True)
        included = []
        skipped = []
        people = self.list_people(self.SOURCE_DIR)
        for index, name in enumerate(people, start=1):
            self.progress(f"Selecting images ({index}/{len(people)}): {name}")
            images = list_images(os.path.join(self.SOURCE_DIR, name))
            if len(images) < self.MIN_IMAGES:
                skipped.append((name, len(images)))
                continue
            selected = images[:self.MIN_IMAGES]
            prediction_dest = os.path.join(self.PREDICTION_DIR, name)
            rescale_dest = os.path.join(self.RESCALE_DIR, name)
            os.makedirs(prediction_dest, exist_ok=True)
            os.makedirs(rescale_dest, exist_ok=True)
            for path in selected[:self.PREDICTION_IMAGES]:
                shutil.copy2(path, prediction_dest)
            for path in selected[self.PREDICTION_IMAGES:]:
                shutil.copy2(path, rescale_dest)
            included.append(name)
        return included, skipped

    def split_dataset(self):
        from sklearn.model_selection import train_test_split

        for directory in (self.TRAIN_DIR, self.TEST_DIR):
            os.makedirs(directory, exist_ok=True)
        for name in self.list_people(self.RESCALE_DIR):
            images = list_images(os.path.join(self.RESCALE_DIR, name))
            if len(images) < 2:
                train_paths, test_paths = images, []
            else:
                train_paths, test_paths = train_test_split(
                    images,
                    test_size=self.TEST_SIZE,
                    random_state=self.RANDOM_STATE,
                )
            train_dest = os.path.join(self.TRAIN_DIR, name)
            test_dest = os.path.join(self.TEST_DIR, name)
            os.makedirs(train_dest, exist_ok=True)
            os.makedirs(test_dest, exist_ok=True)
            for path in train_paths:
                shutil.copy2(path, train_dest)
            for path in test_paths:
                shutil.copy2(path, test_dest)

    def crop_faces(self):
        dropped = 0
        for split_name, split_dir in (("training", self.TRAIN_DIR), ("testing", self.TEST_DIR)):
            people = self.list_people(split_dir)
            for index, name in enumerate(people, start=1):
                self.progress(f"Cropping {split_name} faces ({index}/{len(people)}): {name}")
                for path in list_images(os.path.join(split_dir, name)):
                    image = read_image(path)
                    face = prepare_face(image)
                    if face is None:
                        os.remove(path)
                        dropped += 1
                    elif face is not image and not write_image(path, face):
                        os.remove(path)
                        dropped += 1
        return dropped

    def visualize(self):
        import matplotlib.pyplot as plt

        names = self.list_people(self.SOURCE_DIR)
        counts = [len(list_images(os.path.join(self.SOURCE_DIR, name))) for name in names]
        plt.figure(figsize=(max(8, len(names) * 0.4), 6))
        plt.title("Distribution of number of pictures")
        plt.xlabel("Names")
        plt.ylabel("Number of pictures")
        bars = plt.bar(names, counts)
        plt.xticks(rotation=90)
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    summary = PreProcess().run()
    print(f"Users included: {len(summary['included'])}")
    for name, count in summary["skipped"]:
        print(f"Skipped {name}: {count} images (minimum {PreProcess.MIN_IMAGES})")
    print(f"Images without a detectable face removed: {summary['dropped']}")
