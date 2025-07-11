import cv2
import os
import time ##thinking to use either time or datetime
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import database

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def verificationImages(img):
    ## bLurriness
    blurValue = int(cv2.Laplacian(img, cv2.CV_64F).var())
    if blurValue <  30:
        print('Image is too blur')
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### multiple faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6)

    if len(faces) != 1:
        print('More than one face')
        return False

     ### Face size check
    x, y, w, h = faces[0]
    face_area = w * h
    image_height, image_width = img.shape[:2]
    frame_area = image_width * image_height
    ratio = face_area / frame_area
    if ratio < 0.02 or ratio > 0.8:
        print('Face too small')
        return False

    return True

def add_new_user():
    name = input("Enter new user name: ").strip()
    department = input ("Enter user's department: ")

    user_dir = os.path.join("Data\\Train", name)
    if os.path.exists(user_dir):
        print(f"User '{name}' already exists.")
        return
    print(f"Created directory: {user_dir}")

    print("Starting capture. Press ESC or wait for 100 images.")

    if(process_camera('capture', name, user_dir)):
        os.makedirs(user_dir)
        database.insert_user(database.get_connection(), name, department, user_dir)

def scan_and_identify():
    while True:
        print("\nMenu:")
        print("1. Enter")
        print("2. Exit")
        print("0. Quit")
        choice = input("Choose: ").lower().strip()
        if choice == '1':
            if(process_camera(mode = "identify", status= "Enter")):
                print("The entry data is recorded")
        elif choice == '2':
            if(process_camera(mode = "identify", status= "Exit")):
                print("The exit data is recorded")
        elif choice == '0':
            break
        else:
            print("Invalid choice, try again.")

 
# def delete_data(user_dir):
#     if os.path.exists(user_dir):
#         for file in os.listdir(user_dir):
#             file_path = os.path.join(user_dir, file)
#             if os.path.isfile(file_path):
#                 os.remove(file_path)
#         os.rmdir(user_dir)
#         return True
#     return False

def process_camera(mode, name=None, user_dir=None, status=None):
    cam = cv2.VideoCapture(0)
    count = 0
    max_count = 100 if mode == 'capture' else 10
    start_time = time.time()
    result = []
    resultNum = [0] * max_count
    
    if mode == 'identify':
        faces, labels = preprocessing()
        best_k = find_best_k(faces, labels)
        X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(X_train, y_train)  # use X_train instead of faces
        y_test_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
    
    while count < max_count and time.time() - start_time < 30:
        ret, frame = cam.read()
        if not ret:
            cam.release()
            cv2.destroyAllWindows()
            return False
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in face_detections:
            face_crop = frame[y:y+h, x:x+w]
            
            if mode == 'capture':  
                if verificationImages(frame):
                    cv2.imwrite(f"Data\\Train\\{name}\\{name}_{count}.jpg", cv2.resize(face_crop, (250, 250)))
                    count += 1
                    cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 2)

            else:  # identify
                if verificationImages(frame):
                    resized = cv2.resize(face_crop, (50, 50))
                    flat_face = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).flatten().reshape(1, -1)
                    y_pred = knn.predict(flat_face)[0] 
                    
                    # set label to display
                    name = y_pred

                    cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 2)
                    cv2.putText(frame, str(name), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            
                    cv2.imwrite(f"Data\\Anchor\\{name}\\{name}_{count}.jpg", cv2.resize(face_crop, (50, 50)))
                    count += 1
    
                    if name not in result: 
                        result.append(f"{name}")   
                        resultNum.append(1) 
                    else: 
                        index = result.index(name)
                        resultNum[index] += 1

                                    
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == 27:
            cam.release()
            cv2.destroyAllWindows()
            return False
    
    
    cam.release()
    cv2.destroyAllWindows()

    if mode == 'identify':
        if result and resultNum:
            max_index = resultNum.index(max(resultNum)) 
            name = result[max_index] 
            if database.user_exists(database.get_connection(), name):
            
                if status == 'Enter':
                    if database.unfinished_attendance(database.get_connection(), name):
                        print("You must check out before checking in again....")
                    else:
                        database.insert_attendance(database.get_connection(),  datetime.now(),name)
                        
                else: ##Exit 
                    database.update_attendance(database.get_connection(), datetime.now(), name)

                print(f"\nFinal Model Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_test_pred))
            else:
                print("User does not exist. Please register first.")
                return False
        else:
            print("No faces were detected or identified during the session.")
            return False

    return True

def preprocessing(): ##is data collection (not preprocessing)
    base_path = 'Data\\Train'
    faces = []
    labels = []

    for person_name in os.listdir(base_path):
        person_path = os.path.join(base_path, person_name)
        if not os.path.isdir(person_path):
            continue  # skip files

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # skip unreadable files

            resized = cv2.resize(img, (50, 50))
            faces.append(resized.flatten())
            labels.append(person_name)

    return np.asarray(faces), np.asarray(labels)

def find_best_k(X, y, max_k=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_k = 2
    best_acc = 0
    for k in range(2, max_k+1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"K={k}, Accuracy={acc:.2f}")
        if acc > best_acc:
            best_acc = acc
            best_k = k
    print(f"Best K: {best_k} with accuracy: {best_acc:.2f}")
    return best_k

os.makedirs("Data\\Train", exist_ok=True)
os.makedirs("Data\\Anchor", exist_ok=True)

def main():
    database.intialise_database(database.get_connection())

    while True:
        print("\nMenu:")
        print("1. Add new user")
        print("2. Identify User")
        print("0. Quit")
        choice = input("Choose: ").lower().strip()
        if choice == '1':
            add_new_user()
        elif choice == '2':
            scan_and_identify()
        elif choice == '0':
            break
        else:
            print("Invalid choice, try again.")

if __name__ == '__main__':
    main()