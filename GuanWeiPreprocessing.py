# Preprocess Dataset (./Data/Train), cropping faces, recoloring to RGB
import os, stat, shutil
import cv2 as cv
from sklearn.model_selection import train_test_split
from glob import glob
import random
import stat
# import easygui as eg
# import winsound

def handler(func, path, exc_info): 
    os.chmod(path, stat.S_IWRITE)
    func(path)

def Rescalling():

    SOURCE = ("Datasets")
    DESTINATION = ("Rescale_Datasets")
    numberOfPics = []

    if os.path.exists(DESTINATION):
        shutil.rmtree(DESTINATION, onerror=handler)


    for name in os.listdir(SOURCE):
        numberOfPics.append(len(os.listdir(os.path.join(SOURCE, name))))

    # Scale all the pic num to min size
    if os.path.exists(DESTINATION) == False:
        os.makedirs(DESTINATION)

    for name in os.listdir(SOURCE):
        person_source = os.path.join(SOURCE, name)
        person_dest = os.path.join(DESTINATION, name)

        if os.path.exists(os.path.join(DESTINATION, name)) == False:
            os.makedirs(os.path.join(DESTINATION, name))

        images = os.listdir(person_source)


        selected_images = random.sample(images, min(numberOfPics))

        
        for img_name in selected_images:
            shutil.copy(src=os.path.join(person_source, img_name), dst=os.path.join(person_dest, img_name))


def PreProcess(algorithm, TEST_SIZE):
    cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    data_set_path = os.path.join(os.getcwd(), 'Rescale_Datasets')

    # Get the name
    username = [x for x in os.listdir(data_set_path)]


    #remove Data if exists
    if os.path.exists("Data"):
        shutil.rmtree("Data", onerror=handler)


    # Split the datasets
    SOURCE_DIR = "Rescale_Datasets"
    DESTINATION_DIR = "Data"
    TRAIN_DIR  = os.path.join(DESTINATION_DIR, "train")
    TEST_DIR   = os.path.join(DESTINATION_DIR, "test")
    TEST_SIZE = TEST_SIZE

    for split_dir in (TRAIN_DIR, TEST_DIR):
        if not os.path.isdir(split_dir):
            os.makedirs(split_dir)

    for person_folder in os.listdir(SOURCE_DIR):
        class_src = os.path.join(SOURCE_DIR, person_folder)
        if not os.path.isdir(class_src):
            continue


        image_paths = glob(os.path.join(class_src, "*.jpg"))
        if not image_paths: 
            continue
        
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
            shutil.copy(src_path, os.path.join(train_dest, os.path.basename(src_path)))
        for src_path in test_paths:
            shutil.copy(src_path, os.path.join(test_dest, os.path.basename(src_path)))
                
        # for src_path in train_paths:
        #     shutil.copy(src_path, train_dest)
        # for src_path in test_paths:
        #     shutil.copy(src_path, test_dest)

    for person_folder in os.listdir(TRAIN_DIR):
        class_src = os.path.join(TRAIN_DIR, person_folder)
        if not os.path.isdir(class_src):
            continue

        train_image_path = glob(os.path.join(class_src, "*.jpg"))
        for path in train_image_path:
            img = cv.imread(path)
            if img is None:
                continue

            grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(grayImg, 1.1, 4)

            if len(faces) == 0:
                continue

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)
            face_crop = img[y:y2, x:x2]
            if face_crop.size == 0:
                continue

            face_crop = cv.cvtColor(face_crop, cv.COLOR_BGR2RGB)
            face_crop = cv.resize(face_crop, (160, 160))

            if face_crop is not None:
                cv.imwrite(path, face_crop)

    for person_folder in os.listdir(TEST_DIR):
        class_src = os.path.join(TEST_DIR, person_folder)
        if not os.path.isdir(class_src):
            continue

        test_image_path = glob(os.path.join(class_src, "*.jpg"))
        for path in test_image_path:
            img = cv.imread(path)
            if img is None:
                continue

            grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(grayImg, 1.1, 4)

            if len(faces) == 0:
                continue

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)
            face_crop = img[y:y2, x:x2]
            if face_crop.size == 0:
                continue

            face_crop = cv.cvtColor(face_crop, cv.COLOR_BGR2RGB)
            face_crop = cv.resize(face_crop, (160, 160))

            if face_crop is not None:
                cv.imwrite(path, face_crop)

    # winsound.MessageBeep()
    # eg.msgbox(msg="Done", title="PreProcessing")

# if __name__ == "__main__":
#     Rescalling()
#     print("Created Rescale_Datasets:", os.listdir("Rescale_Datasets"))
#     PreProcess("facenet", 0.2)


# print(os.path.join(data_set_path, username[0], username[0] + '_' + str(i) + '.jpg')) // Check Path 
# print(len(os.listdir(os.path.join(data_set_path, username[0]))))

# cv.imshow('Face Detection', img)
# cv.waitKey(0)