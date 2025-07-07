##cmake version 4.1.0-rc1 , need download at website,restart cmd
##pip install face_recognition



import cv2
import os
import easygui as eg
import face_recognition
import numpy as np #for comparision

# Initialize face detector and camera
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') #detect frontal face
capture = cv2.VideoCapture(0) #open camera
currentFrame = 0

# Global variables for face database
known_encodings = []
known_names = []

def load_known_faces():
    """Load all known faces from the database"""
    global known_encodings, known_names
    known_encodings = []
    known_names = []
    
    if not os.path.exists('captured_frames'):
        os.makedirs('captured_frames')
        return
    
    for name in os.listdir('captured_frames'):
        person_dir = os.path.join('captured_frames', name)
        if os.path.isdir(person_dir):
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

# Load existing faces at startup
load_known_faces()

# Get user name for new registration
input_name = eg.enterbox("Enter your name (leave empty for recognition only):", "Face Recognition System")
registration_mode = bool(input_name) #empty enter will be false , if got key in name will be true

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
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        
        # Register new face
        if registration_mode and currentFrame < 100:  # Capture 20 samples
            cv2.imwrite(f'captured_frames/{input_name}/frame_{currentFrame}.jpg', face_img)
            currentFrame += 1
            cv2.putText(frame, f"Capturing: {currentFrame}/20", (x, y-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Recognize face
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_face)
        #if face detected
        if face_encodings:
            # Compare with known faces,face_encoding[0]is first face that detected,maybe one picture will detected multiple face
            matches = face_recognition.compare_faces(known_encodings, face_encodings[0], tolerance=0.5)
            #smaller distance = more similar
            face_distances = face_recognition.face_distance(known_encodings, face_encodings[0])
            
            if True in matches:
                #find the best match (argmin)
                best_match_index = np.argmin(face_distances)
                name = known_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                #appear of name that detected
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Face Recognition System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final training if in registration mode
if registration_mode:
    load_known_faces()  # Reload database with new faces
    print(f"Registration complete! Added {currentFrame} samples for {input_name}")

capture.release()
cv2.destroyAllWindows()