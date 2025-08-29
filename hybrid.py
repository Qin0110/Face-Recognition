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
# Add Preprocess folder to sys.path
sys.path.append(r"C:\Users\Lenovo\OneDrive\Desktop\Preprocess")
from hybrid_evaluation import evaluation_dateset_face

# === Initialize Haar Cascade for face detection ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 

# Global variables for face database
known_names = []
known_encodings = []
train_names = []
train_encodings = []

# === Cosine Similarity ===
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# === Euclidean Distance ===
def euclidean_distance(vec1, vec2):
    return norm(vec1 - vec2)


# Auto gamma adjustment based on brightness
def adjust_gamma_auto(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    # Decide gamma based on brightness
    if mean_brightness < 80:      # Very dark
        gamma = 1.8
    elif mean_brightness < 120:   # Slightly dark
        gamma = 1.4
    elif mean_brightness > 200:   # Too bright
        gamma = 0.8
    else:                         # Normal brightness
        gamma = 1.0

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table), gamma

# # --- Split Dataset ---
# def split_dataset(source_dir, dest_dir, test_size=0.2):
#     TRAIN_DIR = os.path.join(dest_dir, "train")
#     TEST_DIR = os.path.join(dest_dir, "test")
#     if os.path.exists(dest_dir):
#         shutil.rmtree(dest_dir)
#     os.makedirs(TRAIN_DIR, exist_ok=True)
#     os.makedirs(TEST_DIR, exist_ok=True)

#     for person_name in os.listdir(source_dir):
#         person_folder = os.path.join(source_dir, person_name)
#         if not os.path.isdir(person_folder):
#             continue
#         image_paths = glob(os.path.join(person_folder, "*.jpg")) + \
#                       glob(os.path.join(person_folder, "*.jpeg")) + \
#                       glob(os.path.join(person_folder, "*.png"))
#         if not image_paths: continue
#         train_paths, test_paths = train_test_split(image_paths, test_size=test_size, random_state=42)
#         os.makedirs(os.path.join(TRAIN_DIR, person_name), exist_ok=True)
#         os.makedirs(os.path.join(TEST_DIR, person_name), exist_ok=True)
#         for path in train_paths: shutil.copy(path, os.path.join(TRAIN_DIR, person_name, os.path.basename(path)))
#         for path in test_paths: shutil.copy(path, os.path.join(TEST_DIR, person_name, os.path.basename(path)))
#     print(f"Dataset split done. Train: {TRAIN_DIR}, Test: {TEST_DIR}")
 
# --- Accuracy Calculation ---
def calc_metrics(tp, tn, fp, fn):
    precision = tp / (tp + fp)if tp + fp else 0 # if tp + fp not equal to 0 then calc
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0
    return precision, recall, f1, accuracy

# --- Train dataset
def train_dateset_face(train_dir):
    # --- Train ---
    #loop all person in Train file
    for person_name in os.listdir(train_dir):
        #combine file path of folder name "Train" and folder name path
        person_folder = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_folder): 
            continue
        person_encodings = [] #store all encoding of this person
        print(f"===========================================")
        #add their file path(that people of all picture)
        for image_file in os.listdir(person_folder):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')): 
                continue
            #combine file path of the folder name path and image file 
            image_path = os.path.join(person_folder, image_file)
            try:
                # Load image using OpenCV
                image_cv = cv2.imread(image_path)
                if image_cv is None: 
                    print(f"Could not read {image_path}")
                    continue

                # Auto gamma adjustment
                image_cv, gamma_used = adjust_gamma_auto(image_cv)
                print(f"adjusted gamma for : {person_name} -- {image_file}: gamma applied = {gamma_used}")
                
                # Convert BGR to RGB for face_recognition
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

                # Compute face encodings
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


# def evaluation_dateset_face(test_dir):
#     train_names,train_encodings = load_dataset_faces_from_json()
#     if len(train_names) == 0 and len(train_encodings) == 0:
#         print("No data found in JSON.")
#         eg.buttonbox(
#             msg="⚠ No known faces found!\n\nPlease run 'Train Dataset Model' first.",
#             title="Warning",
#             choices=["OK"]
#         )
#     else:
#         # --- Test with TP, TN, FP, FN ---
#         tp = tn = fp = fn = 0
#         for person_name in os.listdir(test_dir):
#             person_folder = os.path.join(test_dir, person_name)
#             if not os.path.isdir(person_folder): 
#                 continue
#             for image_file in os.listdir(person_folder):
#                 if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')): 
#                     continue
#                 image_path = os.path.join(person_folder, image_file)
#                 image_cv = cv2.imread(image_path)
#                 if image_cv is None: 
#                     fn += 1
#                     continue
#                 image_cv, _ = adjust_gamma_auto(image_cv)
#                 image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
#                 # detect face location in the image
#                 face_locations = face_recognition.face_locations(image_rgb)
#                 encodings = face_recognition.face_encodings(image_rgb, face_locations)

#                 if encodings:
#                     encoding = encodings[0]
#                     best_name = "Unknown"
#                     best_similarity = 0
#                     # Compare with all known faces using hybrid check
#                     for idx, known in enumerate(train_encodings):
#                         cos_sim = cosine_similarity(encoding, known)
#                         euc_dist = euclidean_distance(encoding, known)
#                         if cos_sim > 0.7 and euc_dist < 0.4:
#                             if cos_sim > best_similarity:
#                                 best_similarity = cos_sim
#                                 best_name = train_names[idx]

#                     # Update metrics
#                     # predicted the correct person
#                     if best_name == person_name:
#                         tp += 1
#                     # it can't recognize the person that should be recognizes
#                     elif best_name == "Unknown":
#                         fn += 1
#                     # it can recognize the person as someone, but it recognize wrong name
#                     else:
#                         fp += 1
#                 # cannot do encoding also considered as fn because expect recognition but failed
#                 else:
#                     fn += 1

#         tn = 0  # For multi-class, TN is less meaningful here

#         precision, recall, f1, accuracy = calc_metrics(tp, tn, fp, fn)
#         print(f"TP: {tp}, FP: {fp}, FN: {fn}")
#         print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}")

#         # --- Metrics Graph ---
#         metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
#         values = [accuracy, precision, recall, f1]
#         plt.figure(figsize=(6,4))
#         plt.bar(metrics, values, color=['skyblue', 'orange', 'green', 'red'])
#         plt.ylim(0, 1)
#         plt.title("Model Performance Metrics")
#         plt.ylabel("Score")
#         plt.show()

#         # --- Confusion Matrix ---
#         conf_matrix = np.array([[tp, fn],
#                                 [fp, tn]])
#         plt.figure(figsize=(5,4))
#         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#                     xticklabels=['Predicted Positive', 'Predicted Negative'],
#                     yticklabels=['Actual Positive', 'Actual Negative'])
#         plt.title("Confusion Matrix")
#         plt.show()

# Load Known Faces with Averaging and Auto Gamma (train the model)
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
                person_encodings = []  # Store all encodings for this person
                print(f"===========================================")
                for image_file in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_file)
                    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    try:
                        # Load image using OpenCV
                        image_cv = cv2.imread(image_path)
                        if image_cv is None:
                            print(f"Could not read {image_path}")
                            continue

                        # Auto gamma adjustment
                        image_cv, gamma_used = adjust_gamma_auto(image_cv)
                        print(f"adjusted gamma for : {name} -- {image_file}: gamma applied = {gamma_used}")
                        # Convert BGR to RGB for face_recognition
                        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

                        # Compute face encodings
                        encodings = face_recognition.face_encodings(image_rgb)
                        if encodings:
                            person_encodings.append(encodings[0])
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

                # Average embeddings for this person
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
            "encoding": encoding.tolist()  # Convert NumPy array to list
        })

    with open(filepath, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data)} known faces to {filepath}")

def load_known_faces_from_json(filepath="./Data/known_face.json"):
    # Check if file exists and is not empty
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        print("JSON file does not exist or is empty.")
        return [], []  # Return empty lists

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
        known_encodings.append(np.array(item["encoding"]))  # Convert list back to NumPy array

    return known_names, known_encodings

def save_dataset_faces_to_json(train_names,train_encodings,filepath="./Data/dataset_face.json"):
    data = []
    for name, encoding in zip(train_names, train_encodings):
        data.append({
            "name": name,
            "encoding": encoding.tolist()  # Convert NumPy array to list
        })

    with open(filepath, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data)} known faces to {filepath}")

def load_dataset_faces_from_json(filepath="./Data/dataset_face.json"):
    # Check if file exists and is not empty
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        print("JSON file does not exist or is empty.")
        return [], []  # Return empty lists

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
        known_encodings.append(np.array(item["encoding"]))  # Convert list back to NumPy array

    return known_names, known_encodings



# === Register New Face ===
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
            scaleFactor=1.1,       # Keep small increment for better accuracy
            minNeighbors=8,        # Higher value = fewer false positives
            minSize=(120, 120),    # Ignore very small objects
            maxSize=(400, 400),    # (Optional) Ignore overly large objects
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
    # load_known_faces_with_preprocessing()
    print(f"Registration complete! Added {currentFrame} samples for {input_name}")
    eg.msgbox("Registration complete!", "Done")


# === Recognize Faces (Haar + ResNet + Hybrid Check) ===
def recognize_mode():
    known_names,known_encodings = load_known_faces_from_json()
    capture = cv2.VideoCapture(0)
    print("Recognition mode: Haar Detection + ResNet + Hybrid (Cosine + Euclidean). Press ESC to stop.")

    while True:
        ret, frame = capture.read()
        if not ret:
            continue

        # Convert frames
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using Haar
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
                # Compare with all known faces using hybrid check
                for idx, known in enumerate(known_encodings):
                    cos_sim = cosine_similarity(face_encoding, known)
                    euc_dist = euclidean_distance(face_encoding, known)

                    # Hybrid rule: both conditions must pass
                    if cos_sim > 0.7 and euc_dist < 0.4:
                        if cos_sim > best_similarity:  # pick highest cosine similarity
                            best_similarity = cos_sim
                            best_name = known_names[idx]

                # Display result
                if best_name != "Unknown":
                    confidence = best_similarity * 100
                    label = f"{best_name} ({confidence:.1f}%)"
                    color = (0, 255, 0)
                else:
                    label = "Unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Recognition (Haar + hybrid)', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
            break

    capture.release()
    cv2.destroyAllWindows()
    eg.msgbox("Recognition ended.", "Done")


# # === Recognize Specific Person (Hybrid Check: Cosine + Euclidean) ===
# def recognize_specific_person():
#     target_name = eg.enterbox("Enter the name of the person to recognize:", "Specific Recognition")
#     if not target_name:
#         return

#     person_folder = os.path.join('captured_frames', target_name)
#     if not os.path.exists(person_folder) or not os.listdir(person_folder):
#         eg.msgbox(f"Invalid name: '{target_name}' not found in database.", "Error")
#         return

#     capture = cv2.VideoCapture(0)
#     print(f"Recognition mode: Looking only for '{target_name}' (Hybrid Check). Press ESC to stop.")

#     while True:
#         ret, frame = capture.read()
#         if not ret:
#             continue

#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Detect faces using Haar
#         faces = face_cascade.detectMultiScale(
#             gray_frame,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(120, 120),
#             maxSize=(400, 400),
#             flags=cv2.CASCADE_SCALE_IMAGE
#         )

#         for (x, y, w, h) in faces:
#             top, right, bottom, left = y, x + w, y + h, x
#             encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])

#             if encodings:
#                 face_encoding = encodings[0]

#                 best_name = "Unknown"
#                 best_similarity = 0
#                 color = (0, 0, 255)

#                 # Compare with all known faces using hybrid check
#                 for idx, known in enumerate(known_encodings):
#                     cos_sim = cosine_similarity(face_encoding, known)
#                     euc_dist = euclidean_distance(face_encoding, known)

#                     # Hybrid condition: both thresholds must pass (Cosine similarity + Euclidean distance)
#                     if cos_sim > 0.965 and euc_dist < 0.4:
#                         if cos_sim > best_similarity:  # keep the highest cosine similarity
#                             best_similarity = cos_sim
#                             best_name = known_names[idx]

#                 # Display result
#                 if best_name == target_name:
#                     confidence = best_similarity * 100
#                     label = f"{best_name} ({confidence:.1f}%)"
#                     color = (0, 255, 0)
#                 elif best_name != "Unknown":
#                     label = "Not Target"
#                     color = (0, 0, 255)
#                 else:
#                     label = "Unknown"
#                     color = (0, 0, 255)

#                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                 cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         cv2.imshow(f"Recognition: {target_name}", frame)
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     capture.release()
#     cv2.destroyAllWindows()
#     eg.msgbox(f"Specific recognition for '{target_name}' ended.", "Done")

# === Main Menu ===


train_folder =  r'C:\Users\Lenovo\OneDrive\Desktop\Ai assignment (face recognition)\Face-Recognition\Data\Train'
test_folder  = r'C:\Users\Lenovo\OneDrive\Desktop\Ai assignment (face recognition)\Face-Recognition\Data\Prediction'


while True:
  

    mode = eg.buttonbox("Choose mode", "Face Recognition System",
                        choices=["Register", "Recognize (Webcam)", "Train model", "Train dataset model" ,"Evaluate dateset",  "Exit"])
    if mode == "Register":
        register_mode()  
    elif mode == "Recognize (Webcam)":
        recognize_mode()  # webcam recognition
    elif mode == "Train model":
        load_known_faces_with_preprocessing() 
    elif mode == "Train dataset model":
        train_dateset_face(train_folder)
    elif mode == "Evaluate dateset":
        # split_dataset(source_dataset, destination_data, test_size=0.2)
        evaluation_dateset_face(test_folder)
    elif mode == "Exit" or mode is None:
        break
