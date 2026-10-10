

import cv2
import os
import easygui as eg
import face_recognition
import numpy as np

face_cascade = cv2.CascadeClassifier('Face-Recognition/haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
currentFrame = 0

known_encodings = []
known_names = []

def load_known_faces():
    """Load all known faces from the captured_frames and external training folder"""
    global known_encodings, known_names
    known_encodings = []
    known_names = []

    folders_to_scan = [
        'captured_frames',
        r'C:\Users\Lenovo\OneDrive\Desktop\Ai assignment (face recognition)\Face-Recognition\Data\Train'
    ]

    for base_dir in folders_to_scan:
        if not os.path.exists(base_dir):
            continue

        for name in os.listdir(base_dir):
            person_dir = os.path.join(base_dir, name)
            if os.path.isdir(person_dir):
                for image_file in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_file)
                    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            known_encodings.append(encodings[0])
                            known_names.append(name)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

load_known_faces()

input_name = eg.enterbox("Enter your name (leave empty for recognition only):", "Face Recognition System")
registration_mode = bool(input_name)

if registration_mode:
    os.makedirs(f'captured_frames/{input_name}', exist_ok=True)
    print(f"Registration mode: Capturing samples for {input_name}")
else:
    print("Recognition mode: Identifying faces against known database")

while True:
    ret, frame = capture.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        face_img = frame[y:y+h, x:x+w]
        
        if registration_mode and currentFrame < 20:
            cv2.imwrite(f'captured_frames/{input_name}/frame_{currentFrame}.jpg', face_img)
            currentFrame += 1
            cv2.putText(frame, f"Capturing: {currentFrame}/20", (x, y-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_face)
        if not registration_mode and face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encodings[0], tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, face_encodings[0])
            
            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = known_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Face Recognition System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if registration_mode:
    load_known_faces()
    print(f"Registration complete! Added {currentFrame} samples for {input_name}")

capture.release()
cv2.destroyAllWindows()
