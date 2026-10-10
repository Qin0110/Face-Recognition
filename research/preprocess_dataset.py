import os
import shutil
import cv2 as cv
from sklearn.model_selection import train_test_split
from glob import glob
import easygui as eg
import winsound

def PreProcess():
    cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    data_set_path = os.path.join(os.getcwd(), 'Datasets')
    print("Dataset path:", data_set_path)

    username = [x for x in os.listdir(data_set_path)]
    print("Dataset user: ", len(username), "users found.")
    print(username)

    SOURCE_DIR = "Datasets"
    DESTINATION_DIR = "Data"
    TRAIN_DIR  = os.path.join(DESTINATION_DIR, "train")
    TEST_DIR   = os.path.join(DESTINATION_DIR, "test")
    TEST_SIZE = 0.3

    for split_dir in (TRAIN_DIR, TEST_DIR):
        if not os.path.isdir(split_dir):
            os.makedirs(split_dir)

    for person_folder in os.listdir(SOURCE_DIR):
        class_src = os.path.join(SOURCE_DIR, person_folder)
        if not os.path.isdir(class_src):
            continue

        image_paths = glob(os.path.join(class_src, "*.jpg"))
        train_paths, test_paths = train_test_split(
            image_paths,
            test_size = TEST_SIZE,
            random_state = 42,
            stratify = None,
        )

        train_dest = os.path.join(TRAIN_DIR, person_folder)
        test_dest  = os.path.join(TEST_DIR,  person_folder)
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(test_dest,  exist_ok=True)
                
        for src_path in train_paths:
            shutil.copy(src_path, train_dest)
        for src_path in test_paths:
            shutil.copy(src_path, test_dest)

    for person_folder in os.listdir(TRAIN_DIR):
        class_src = os.path.join(TRAIN_DIR, person_folder)
        if not os.path.isdir(class_src):
            continue
        train_image_path = glob(os.path.join(class_src, "*.jpg"))
        for path in train_image_path:
            img = cv.imread(path)
            for(x, y, w, h) in cascade.detectMultiScale(img, 1.1, 4):
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                facedata = img[y:y + h, x:x + w]
                converter = cv.cvtColor(facedata, cv.COLOR_BGR2RGB)
            if converter is not None:
                cv.imwrite(path, converter)

    winsound.MessageBeep()
    eg.msgbox(msg="Done", title="PreProcessing")

if __name__ == "__main__":
    PreProcess()
