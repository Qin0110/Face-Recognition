import cv2
import os
import easygui as eg
import face_recognition
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from numpy import dot
# === Initialize Haar Cascade for face detection ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 



# Global variables for face database
known_encodings = []
known_names = []


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

# Load Known Faces with Averaging and Auto Gamma 
def load_known_faces():
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






# === Evaluate on Test Dataset ===
def evaluate_faces(test_dir):
    y_true, y_pred = [], []

    for person_name in os.listdir(test_dir):
        person_dir = os.path.join(test_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for image_file in os.listdir(person_dir):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_path = os.path.join(person_dir, image_file)
            y_true.append(person_name)

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                encoding = encodings[0]
                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_encodings, encoding)

                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    predicted_name = known_names[best_match_index]
                else:
                    predicted_name = "Unknown"
            else:
                predicted_name = "NoFace"

            y_pred.append(predicted_name)

    print("=== Evaluation Results ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    eg.msgbox("Evaluation complete. Check console for results.", "Done")


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
    load_known_faces()
    print(f"Registration complete! Added {currentFrame} samples for {input_name}")
    eg.msgbox("Registration complete!", "Done")


# === Recognize Faces (Haar + ResNet + Hybrid Check) ===
def recognize_mode():
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
                    if cos_sim > 0.965 and euc_dist < 0.4:
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

        cv2.imshow('Recognition (Haar + ResNet + Hybrid)', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
            break

    capture.release()
    cv2.destroyAllWindows()
    eg.msgbox("Recognition ended.", "Done")


# === Recognize Specific Person (Hybrid Check: Cosine + Euclidean) ===
def recognize_specific_person():
    target_name = eg.enterbox("Enter the name of the person to recognize:", "Specific Recognition")
    if not target_name:
        return

    person_folder = os.path.join('captured_frames', target_name)
    if not os.path.exists(person_folder) or not os.listdir(person_folder):
        eg.msgbox(f"Invalid name: '{target_name}' not found in database.", "Error")
        return

    capture = cv2.VideoCapture(0)
    print(f"Recognition mode: Looking only for '{target_name}' (Hybrid Check). Press ESC to stop.")

    while True:
        ret, frame = capture.read()
        if not ret:
            continue

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

                    # Hybrid condition: both thresholds must pass (Cosine similarity + Euclidean distance)
                    if cos_sim > 0.965 and euc_dist < 0.4:
                        if cos_sim > best_similarity:  # keep the highest cosine similarity
                            best_similarity = cos_sim
                            best_name = known_names[idx]

                # Display result
                if best_name == target_name:
                    confidence = best_similarity * 100
                    label = f"{best_name} ({confidence:.1f}%)"
                    color = (0, 255, 0)
                elif best_name != "Unknown":
                    label = "Not Target"
                    color = (0, 0, 255)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(f"Recognition: {target_name}", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
    eg.msgbox(f"Specific recognition for '{target_name}' ended.", "Done")

# === Main Menu ===
load_known_faces()
while True:
    mode = eg.buttonbox("Choose mode", "Face Recognition System",
                        choices=["Register", "Recognize (Webcam)", "Recognize Specific Person",
                                 "Evaluate Test Set", "Exit"])
    if mode == "Register":
        register_mode()
    elif mode == "Recognize (Webcam)":
        if not known_encodings:
            load_known_faces()
        recognize_mode()
    elif mode == "Recognize Specific Person":
        if not known_encodings:
            load_known_faces()
        recognize_specific_person()
    elif mode == "Evaluate Test Set":
        test_dir = eg.diropenbox("Select your test dataset folder")
        if test_dir:
            evaluate_faces(test_dir)
    elif mode == "Exit" or mode is None:
        break
