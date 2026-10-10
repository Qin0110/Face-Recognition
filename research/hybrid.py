import cv2
import os
import easygui as eg
import face_recognition
import numpy as np

import time

from numpy.linalg import norm
from numpy import dot

import json

from glob import glob

import sys
sys.path.append(r"C:\Users\Lenovo\OneDrive\Desktop\Preprocess")
from hybrid_evaluation import evaluation_dateset_face

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 

known_names = []
known_encodings = []
train_names = []
train_encodings = []

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def euclidean_distance(vec1, vec2):
    return norm(vec1 - vec2)

def adjust_gamma_auto(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness < 80:
        gamma = 1.8
    elif mean_brightness < 120:
        gamma = 1.4
    elif mean_brightness > 200:
        gamma = 0.8
    else:
        gamma = 1.0

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table), gamma

 
def calc_metrics(tp, tn, fp, fn):
    precision = tp / (tp + fp)if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0
    return precision, recall, f1, accuracy

def train_dateset_face(train_dir):
    for person_name in os.listdir(train_dir):
        person_folder = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_folder): 
            continue
        person_encodings = []
        print(f"===========================================")
        for image_file in os.listdir(person_folder):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')): 
                continue
            image_path = os.path.join(person_folder, image_file)
            try:
                image_cv = cv2.imread(image_path)
                if image_cv is None: 
                    print(f"Could not read {image_path}")
                    continue

                image_cv, gamma_used = adjust_gamma_auto(image_cv)
                print(f"adjusted gamma for : {person_name} -- {image_file}: gamma applied = {gamma_used}")
                
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

                encodings = face_recognition.face_encodings(image_rgb)
                if encodings: 
                    person_encodings.append(encodings[0])
            except Exception as e:
                        print(f"Error processing {image_path}: {e}")

        if person_encodings:
            avg_encoding = np.mean(person_encodings, axis=0)
            train_encodings.append(avg_encoding)
            train_names.append(person_name)
            print(f"Trained {person_name} with {len(person_encodings)} images")
    save_dataset_faces_to_json(train_names,train_encodings)

def load_known_faces_with_preprocessing():
    
    global known_encodings, known_names
    known_encodings = []
    known_names = []

    folders_to_scan = ['captured_frames']

    for base_dir in folders_to_scan:
        if not os.path.exists(base_dir):
            continue
        for name in os.listdir(base_dir):
            person_dir = os.path.join(base_dir, name)
            if os.path.isdir(person_dir):
                person_encodings = []
                print(f"===========================================")
                for image_file in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_file)
                    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    try:
                        image_cv = cv2.imread(image_path)
                        if image_cv is None:
                            print(f"Could not read {image_path}")
                            continue

                        image_cv, gamma_used = adjust_gamma_auto(image_cv)
                        print(f"adjusted gamma for : {name} -- {image_file}: gamma applied = {gamma_used}")
                        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

                        encodings = face_recognition.face_encodings(image_rgb)
                        if encodings:
                            person_encodings.append(encodings[0])
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

                if person_encodings:
                    avg_encoding = np.mean(person_encodings, axis=0)
                    known_encodings.append(avg_encoding)
                    known_names.append(name)
                    print(f"Loaded and averaged {len(person_encodings)} embeddings for {name}")
    save_known_faces_to_json()
   
def save_known_faces_to_json(filepath="./Data/known_face.json"):
    data = []
    for name, encoding in zip(known_names, known_encodings):
        data.append({
            "name": name,
            "encoding": encoding.tolist()
        })

    with open(filepath, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data)} known faces to {filepath}")

def load_known_faces_from_json(filepath="./Data/known_face.json"):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        print("JSON file does not exist or is empty.")
        return [], []

    with open(filepath, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("JSON file is invalid (does not follow JSON format rules )).")
            return [], []

    if not data:
        print("No data found in JSON file.")
        return [], []

    known_names = []
    known_encodings = []

    for item in data:
        known_names.append(item["name"])
        known_encodings.append(np.array(item["encoding"]))

    return known_names, known_encodings

def save_dataset_faces_to_json(train_names,train_encodings,filepath="./Data/dataset_face.json"):
    data = []
    for name, encoding in zip(train_names, train_encodings):
        data.append({
            "name": name,
            "encoding": encoding.tolist()
        })

    with open(filepath, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data)} known faces to {filepath}")

def load_dataset_faces_from_json(filepath="./Data/dataset_face.json"):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        print("JSON file does not exist or is empty.")
        return [], []

    with open(filepath, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("JSON file is invalid (does not follow JSON format rules )).")
            return [], []

    if not data:
        print("No data found in JSON file.")
        return [], []

    with open(filepath, "r") as f:
        data = json.load(f)

    known_names = []
    known_encodings = []

    for item in data:
        known_names.append(item["name"])
        known_encodings.append(np.array(item["encoding"]))

    return known_names, known_encodings

def register_mode():
    capture = cv2.VideoCapture(0)
    input_name = eg.enterbox("Enter your name:", "Registration")
    if not input_name:
        capture.release()
        cv2.destroyAllWindows()
        return
    os.makedirs(f'captured_frames/{input_name}', exist_ok=True)
    print(f"Registration mode: Capturing samples for {input_name}")

    currentFrame = 0
    while currentFrame < 20:
        ret, frame = capture.read()
        if not ret:
            continue
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(120, 120),
            maxSize=(400, 400),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = frame[y:y + h, x:x + w]
            cv2.imwrite(f'captured_frames/{input_name}/frame_{currentFrame}.jpg', face_img)
            currentFrame += 1
            cv2.putText(frame, f"Capturing: {currentFrame}/20", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            time.sleep(0.3)

        cv2.imshow('Registration', frame)
        cv2.waitKey(1)

    capture.release()
    cv2.destroyAllWindows()
    print(f"Registration complete! Added {currentFrame} samples for {input_name}")
    eg.msgbox("Registration complete!", "Done")

def recognize_mode():
    known_names,known_encodings = load_known_faces_from_json()
    capture = cv2.VideoCapture(0)
    print("Recognition mode: Haar Detection + ResNet + Hybrid (Cosine + Euclidean). Press ESC to stop.")

    while True:
        ret, frame = capture.read()
        if not ret:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(120, 120),
            maxSize=(400, 400),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            top, right, bottom, left = y, x + w, y + h, x
            encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])

            if encodings:
                face_encoding = encodings[0]
                best_name = "Unknown"
                best_similarity = 0
                color = (0, 0, 255)
                for idx, known in enumerate(known_encodings):
                    cos_sim = cosine_similarity(face_encoding, known)
                    euc_dist = euclidean_distance(face_encoding, known)

                    if cos_sim > 0.7 and euc_dist < 0.4:
                        if cos_sim > best_similarity:
                            best_similarity = cos_sim
                            best_name = known_names[idx]

                if best_name != "Unknown":
                    confidence = best_similarity * 100
                    label = f"{best_name} ({confidence:.1f}%)"
                    color = (0, 255, 0)
                else:
                    label = "Unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Recognition (Haar + hybrid)', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
    eg.msgbox("Recognition ended.", "Done")

train_folder =  r'C:\Users\Lenovo\OneDrive\Desktop\Ai assignment (face recognition)\Face-Recognition\Data\Train'
test_folder  = r'C:\Users\Lenovo\OneDrive\Desktop\Ai assignment (face recognition)\Face-Recognition\Data\Prediction'

while True:
  

    mode = eg.buttonbox("Choose mode", "Face Recognition System",
                        choices=["Register", "Recognize (Webcam)", "Train model", "Train dataset model" ,"Evaluate dateset",  "Exit"])
    if mode == "Register":
        register_mode()  
    elif mode == "Recognize (Webcam)":
        recognize_mode()
    elif mode == "Train model":
        load_known_faces_with_preprocessing() 
    elif mode == "Train dataset model":
        train_dateset_face(train_folder)
    elif mode == "Evaluate dateset":
        evaluation_dateset_face(test_folder)
    elif mode == "Exit" or mode is None:
        break
