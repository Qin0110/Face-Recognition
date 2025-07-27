import cv2
import os
from datetime import datetime
import numpy as np
import database
from keras_facenet import FaceNet
from PIL import Image, ImageTk

embedder = FaceNet() 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def check_user_exists(user_id):
    return database.user_exists(database.get_connection(), user_id)

def add_new_user(name, department = None, created_by = None):
    
    try:
        user_dir = os.path.join("Data\\Train", name)

        if os.path.exists(user_dir):
            return False, f"User directory already exists for {name}"
            
        os.makedirs(user_dir, exist_ok=True)
        
        isTrue, message = database.insert_user(database.get_connection(), name, department, user_dir, created_by)
        
        if not isTrue:
            if os.path.exists(user_dir):
                os.rmdir(user_dir)
            return False, f"Database error: {message}"
            
        return True, message
        
    except Exception as e:
        if os.path.exists(user_dir):
            os.rmdir(user_dir)
        return False, f"Error adding user: {e}"

def check_admin_exists(admin_id, admin_pass= None):
    return database.admin_exists(database.get_connection(), admin_id, admin_pass)


def add_admin(admin_id, admin_pass):
    return database.insert_admin(database.get_connection(),admin_id, admin_pass)

def display_user():
    return database.display_user(database.get_connection())

def display_user_info(user_id):
    return database.display_user_info(database.get_connection(), user_id)
    

def get_profile_img(user_id, img_base_path):
    for i in range(100):
        trial_path = os.path.join(img_base_path, f"{user_id}_{i}.jpg")
        if os.path.exists(trial_path):
            full_image_path = trial_path
            return full_image_path
        
    return None

def delete_user(user_id, image_path):
    if os.path.isfile(image_path):
        os.remove(image_path)
    return database.delete_user(database.get_connection(), user_id)

def update_user_info(user_id, department_new):
    return database.update_user_data(database.get_connection(), user_id, department_new)






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
                continue

        if user_embedding:

            # average embeddings for each person 
            avg_embedding = np.mean(user_embedding, axis=0)
            embeddings.append(avg_embedding)
            labels.append(user)

    return np.array(embeddings), np.array(labels)

def precompute_embeddings():
    path = 'Data\\Train'
    embeddings_path = 'Data\\embeddings.npz'
    
    
    embeddings, labels = load_embeddings(path)
    
    
    np.savez(embeddings_path, embeddings=embeddings, labels=labels)
    return (f"Saved embeddings for {len(labels)} users")
    

def load_precomputed_embeddings():

    embeddings_path = 'Data\\embeddings.npz'

    if not os.path.exists(embeddings_path):
        return None, None
    
    try:
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        labels = data['labels']
        return embeddings, labels
    except Exception as e:
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
    
    # return match only if distance is below threshold
    if min_distance <= threshold:
        return best_match, min_distance
    else:
        return None, min_distance

def verificationImages(img):

    ## blurriness
    blurValue = int(cv2.Laplacian(img, cv2.CV_64F).var())
    if blurValue < 30:
        return False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### multiple faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) != 1:
        return False

    ### face size check
    x, y, w, h = faces[0]
    face_area = w * h
    image_height, image_width = img.shape[:2]
    frame_area = image_width * image_height
    ratio = face_area / frame_area
    if ratio < 0.02 or ratio > 0.8:
        return False

    return True

def capturing_images(frame, face_crop, name, count):
    if verificationImages(frame):
        cv2.imwrite(f"Data\\Train\\{name}\\{name}_{count}.jpg", cv2.resize(face_crop, (250, 250)))
        return True
    else: 
        return False
    

def take_attendance(result, resultNum, status):
        
    if not result or not resultNum:
        return False, "No faces were detected during the session."

    max_index = resultNum.index(max(resultNum)) 
    final_name = result[max_index]       

    exists, existMsg = check_user_exists(final_name)
    if not exists:
            return False, "User does not exist. Please register first."
    
    if status == 'Enter':
        hasUnfinished, unfinishedMsg = database.unfinished_attendance(database.get_connection(), final_name)
        if hasUnfinished:
            return False, "You must check out before checking in again...."
        
        isInserted, message = database.insert_attendance(database.get_connection(), final_name)

        if isInserted:
            return True, f"Entry recorded for {final_name} successfully"
        else:
            return False, message
                        
    else:  # exit 

        hasUnfinished, unfinishedMsg = database.unfinished_attendance(database.get_connection(), final_name)
        if not hasUnfinished:
                return False, "You must check in before checking out."
        
        isUpdated, updateMsg = database.update_attendance(database.get_connection(), datetime.now(), final_name)

        if isUpdated:
            return True, f"Exit recorded for {final_name} successfully"
        else:
            return False, "You must check in before checking out...."


def close_camera(cam):
    if cam is not None:
        cam.release()
        cv2.destroyAllWindows()

def open_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return None
    
    return cam


def update_camera(cam, mode, status, username, camera_label, page, stored_embeddings, stored_labels):
    result = []
    resultNum = []
    count = 0
    is_updating = True
    max_count = 100 if mode == 'capture' else 10 if mode == 'attendance' else 0

    while (count < max_count if mode == 'capture' else count < max_count if mode == 'attendance' else is_updating):                
        if not is_updating:  
            break
        
        if cam is not None and cam.isOpened():
            isTrue, frame = cam.read()

            if isTrue:
                grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_detection = face_cascade.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=5)       
                if len(face_detection) > 0:
                    cv2.putText(frame, f"Faces detected: {len(face_detection)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No faces detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                for (x, y, w, h) in face_detection:
                    face_crop = frame[y:y+h, x:x+w]

                    if mode == 'capture':
                        if(capturing_images(frame = frame, face_crop = face_crop, name = username, count = count)):
                            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Captured: {count}", (x, y-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            count += 1

                        if count >= 10:  # Stop after 10 captures
                            close_camera(cam)
                            camera_label.grid_remove()
                            is_updating = False
                            break
                        
                        cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (255, 255, 0), 2)

                    if mode == None or mode == 'attendance':
                        if(verificationImages(frame)):
                            try:
                                face_embedding = get_embedding(face_crop)

                                identified_user, distance = identify_face_euclidean(face_embedding, stored_embeddings, stored_labels)
                                
                                if identified_user is not None:
                                    cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 2)
                                    cv2.putText(frame, f"{identified_user} ({distance:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                                    if mode == 'attendance':
                                        count += 1
                                        if identified_user not in result:
                                            result.append(identified_user)
                                            resultNum.append(1)
                                        else: 
                                            index  = result.index(identified_user)
                                            resultNum[index] += 1
                                        
                                else:
                                    cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 0, 255), 2)
                                    cv2.putText(frame, f"Unknown ({distance:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            except Exception as e:
                                cv2.putText(frame, "Error", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, "Face verification failed", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = img.resize((498, 532))  # Resize for better fit
                imgtk = ImageTk.PhotoImage(image=img)

                camera_label.imgtk = imgtk
                camera_label.configure(image=imgtk)

                page.update()
            else:
                break
        else:
            break

    if mode == 'attendance' and count >= max_count:
        isTrue, message = take_attendance(result, resultNum, status)
        return isTrue, message
    return None, None




def intialise_database():
    isTrue, message = database.intialise_database(database.get_connection())

    if isTrue:
        if(not os.path.exists("Data\\Train")):
            os.makedirs("Data\\Train")
            return True, None
    else:
        deleted, error_delete = database.deleteDatabase(database.get_connection())
            
        if deleted:
            return False, message
        else: 
            return False, error_delete
    
    return True, None



