import cv2
import os
import time
from datetime import datetime
import numpy as np
import database
from keras_facenet import FaceNet

embedder = FaceNet()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def get_embedding(face_img):
    # RGB and size (160x160)
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) ##color image(bgr) to rgb
    else:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB) ##gray image to rgb
    
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    
    result = embedder.embeddings(face_img)
    return result[0]  # return 512D embedding

def load_embeddings(path):
    embeddings = []
    labels = []

    for user in os.listdir(path):
        user_path = (path + '\\' + user)
        
        if not os.path.isdir(user_path):
            continue

        user_embedding = []
        
        for image_name in os.listdir(user_path):
            image_path = (user_path + '\\' + image_name)
            img = cv2.imread(image_path)
            if img is None:
                continue

            try:
                embedding = get_embedding(img)
                user_embedding.append(embedding)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        if user_embedding:
            # Average embeddings for each person (or use all of them)
            avg_embedding = np.mean(user_embedding, axis=0)
            embeddings.append(avg_embedding)
            labels.append(user)

    return np.array(embeddings), np.array(labels)

def precompute_embeddings():
    path = 'Data\\Train'
    embeddings_path = 'Data\\embeddings.npz'
    
    
    embeddings, labels = load_embeddings(path)
    
    
    np.savez(embeddings_path, embeddings=embeddings, labels=labels)
    print(f"Saved embeddings for {len(labels)} users")
    

def load_precomputed_embeddings():

    embeddings_path = 'Data\\embeddings.npz'

    if not os.path.exists(embeddings_path):
        print("No precomputed embeddings found. Please run 'Precompute embeddings' first.")
        return None, None
    
    try:
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        labels = data['labels']
        print(f"Loaded precomputed embeddings for {len(labels)} users")
        return embeddings, labels
    except Exception as e:
        print(f"Error loading precomputed embeddings: {e}")
        return None, None

def identify_face_euclidean(face_embedding, stored_embeddings, stored_labels, threshold=0.6):
    if stored_embeddings is None or len(stored_embeddings) == 0:
        return None, float('inf')
    
    min_distance = float('inf')
    best_match = None
    
    for i, stored_embedding in enumerate(stored_embeddings):
        distance = np.linalg.norm(face_embedding - stored_embedding)
        
        if distance < min_distance:
            min_distance = distance
            best_match = stored_labels[i]
    
    # Return match only if distance is below threshold
    if min_distance <= threshold:
        return best_match, min_distance
    else:
        return None, min_distance

def verificationImages(img):

    ## Blurriness
    blurValue = int(cv2.Laplacian(img, cv2.CV_64F).var())
    if blurValue < 30:
        return False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### Multiple faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) != 1:
        return False

    ### Face size check
    x, y, w, h = faces[0]
    face_area = w * h
    image_height, image_width = img.shape[:2]
    frame_area = image_width * image_height
    ratio = face_area / frame_area
    if ratio < 0.02 or ratio > 0.8:
        return False

    return True

def add_new_user():
    name = input("Enter new user name: ").strip()
    department = input("Enter user's department: ")

    user_dir = os.path.join("Data\\Train", name)
    if os.path.exists(user_dir):
        print(f"User '{name}' already exists.")
        return
    
    os.makedirs(user_dir, exist_ok=True)
    print(f"Created directory: {user_dir}")

    print("Starting capture. Press ESC or wait for 100 images.")

    if process_camera('capture', name, user_dir):
        database.insert_user(database.get_connection(), name, department, user_dir)

def scan_and_identify():
    while True:
        print("\nMenu:")
        print("1. Enter")
        print("2. Exit")
        print("0. Quit")
        choice = input("Choose: ").lower().strip()
        if choice == '1':
            if process_camera(mode="identify", status="Enter"):
                print("The entry data is recorded")
        elif choice == '2':
            if process_camera(mode="identify", status="Exit"):
                print("The exit data is recorded")
        elif choice == '0':
            break
        else:
            print("Invalid choice, try again.")

def process_camera(mode, name=None, user_dir=None, status=None):
    cam = cv2.VideoCapture(0)
    count = 0
    max_count = 100 if mode == 'capture' else 5  
    start_time = time.time()
    result = []
    resultNum = []  
    face_detected = False  
    
    if mode == 'identify':
        stored_embeddings, stored_labels = load_precomputed_embeddings()
        if stored_embeddings is None or len(stored_embeddings) == 0:
            print("No precomputed embeddings found. Please run 'Precompute embeddings' first.")
            cam.release()
            cv2.destroyAllWindows()
            return False
    
    print(f"Starting camera for {mode} mode. Press ESC to exit.")
    
    while count < max_count and time.time() - start_time < 30:
        ret, frame = cam.read()
        if not ret:
            cam.release()
            cv2.destroyAllWindows()
            return False
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detection = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # face detection
        if len(face_detection) > 0:
            face_detected = True
            cv2.putText(frame, f"Faces detected: {len(face_detection)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No faces detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for (x, y, w, h) in face_detection:
            face_crop = frame[y:y+h, x:x+w]
            
            if mode == 'capture':  
                if verificationImages(frame):
                    cv2.imwrite(f"Data\\Train\\{name}\\{name}_{count}.jpg", cv2.resize(face_crop, (250, 250)))
                    count += 1
                    cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Captured: {count}", (x, y-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            else:  # identify
                #show face detection box
                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (255, 255, 0), 2)
                
                if verificationImages(frame):
                    print(f"Processing face {count + 1}...")  # Debug info
                    
                    try:
                        # get embedding for current face
                        face_embedding = get_embedding(face_crop)
                        
                        # identify face
                        identified_user, distance = identify_face_euclidean(face_embedding, stored_embeddings, stored_labels)
                        
                        print(f"Similarity: {distance:.3f}")  
                        
                        if identified_user is not None:
                            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{identified_user} ({distance:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            
                            count += 1
                            
                            # track results
                            if identified_user not in result: 
                                result.append(identified_user)   
                                resultNum.append(1) 
                            else: 
                                index = result.index(identified_user)
                                resultNum[index] += 1
                                
                            print(f"Identified: {identified_user} (confidence: {distance:.3f})")
                        else:
                            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 0, 255), 2)
                            cv2.putText(frame, f"Unknown ({distance:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            print(f"Unknown face detected (similarity: {distance:.3f})")
                            
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        cv2.putText(frame, "Error", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Face verification failed", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                                    
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == 27:
            cam.release()
            cv2.destroyAllWindows()
            return False
    
    cam.release()
    cv2.destroyAllWindows()

    if mode == 'identify':
        print(f"\nSession Summary:")
        print(f"Face detected: {face_detected}")
        print(f"Results: {result}")
        print(f"Result counts: {resultNum}")
        
        if result and resultNum:
            max_index = resultNum.index(max(resultNum)) 
            final_name = result[max_index] 
            print(f"Final identified user: {final_name}")
            
            if database.user_exists(database.get_connection(), final_name):
                if status == 'Enter':
                    if database.unfinished_attendance(database.get_connection(), final_name):
                        print("You must check out before checking in again....")
                    else:
                        database.insert_attendance(database.get_connection(), datetime.now(), final_name)
                        print(f"✓ Entry recorded for {final_name}")
                        
                else:  # exit 
                    database.update_attendance(database.get_connection(), datetime.now(), final_name)
                    print(f"✓ Exit recorded for {final_name}")
                    
            else:
                print("User does not exist. Please register first.")
                return False
        else:
            if face_detected:
                print("Faces were detected but none were identified (may be unknown users or low confidence).")
            else:
                print("No faces were detected during the session.")
            return False

    return True


def main():
    database.intialise_database(database.get_connection())
    if(not os.path.exists("Data\\Train")):
        os.makedirs("Data\\Train")

    while True:
        print("\nMenu:")
        print("1. Add new user")
        print("2. Identify User")
        print("3. Training Model")
        print("0. Quit")
        choice = input("Choose: ").lower().strip()
        if choice == '1':
            add_new_user()
        elif choice == '2':
            scan_and_identify()
        elif choice == '3':
            precompute_embeddings()
        elif choice == '0':
            break
        else:
            print("Invalid choice, try again.")

if __name__ == '__main__':
    main()