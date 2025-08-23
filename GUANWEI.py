import cv2
import os
from datetime import datetime
import numpy as np
import database
from PIL import Image, ImageTk
import GuanWeiPreprocessing as preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def check_user_exists(user_id):
    return database.user_exists(database.get_connection(), user_id)

def add_new_user(name, department = None, created_by = None):
    
    try:
        user_dir = os.path.join("Datasets", name)

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
    try:
        # Ensure RGB format
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        else:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)

        # Resize to FaceNet expected size
        face_img = cv2.resize(face_img, (160, 160))

        # Convert to tensor [0,1] → normalize to [-1,1]
        face_tensor = torch.tensor(face_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        face_tensor = (face_tensor - 0.5) / 0.5
        face_tensor = face_tensor.unsqueeze(0).to(device)  # Add batch dim, send to GPU

        # Get embedding
        with torch.no_grad():
            embedding = model(face_tensor).squeeze(0)  # Shape (512,)

        return embedding  # Still on GPU

    except Exception as e:
        raise e

def trainData(path):
    embeddings = []
    labels = []

    for user in os.listdir(path):
        user_path = (path + '\\' + user)
        
        if not os.path.isdir(user_path):
            continue
        
        for image_name in os.listdir(user_path):
            image_path = (user_path + '\\' + image_name)
            img = cv2.imread(image_path)

            if img is None:
                continue

            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                embedding = get_embedding(img)
                embeddings.append(embedding)
                labels.append(user)
                del img

            except Exception as e:
                continue

    embeddings_tensor = torch.stack(embeddings, dim=0)
    return embeddings_tensor, labels

def testData(path, threshold):
    true_positive = 0 
    true_negative = 0
    false_positive = 0
    false_negative = 0

    stored_embeddings, stored_labels = load_precomputed_embeddings()

    if stored_embeddings is None:
        return 0, 0, 0, 0
    
    # safe conversion of 'stored_embeddings' to device float tensor
    if isinstance(stored_embeddings, torch.Tensor):
        stored_embeddings = stored_embeddings.clone().detach().to(device).float()
    elif isinstance(stored_embeddings, np.ndarray):
        stored_embeddings = torch.from_numpy(stored_embeddings.astype('float32')).to(device)
    else:
        # fallback: try to make a numpy array first
        stored_embeddings = torch.tensor(np.array(stored_embeddings, dtype='float32'), dtype=torch.float32, device=device)


    for user in os.listdir(path):
        user_path = (path + '\\' + user)
        
        if not os.path.isdir(user_path):
            continue

        for image_name in os.listdir(user_path):
            image_path = (user_path + '\\' + image_name)

            img = cv2.imread(image_path)

            if img is None:
                continue
            # GrayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # faces = face_cascade.detectMultiScale(GrayScale, scaleFactor=1.1, minNeighbors=4)

            # if len(faces) == 0:
            #     continue

            # x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            # x2 = min(img.shape[1], x + w)
            # y2 = min(img.shape[0], y + h)
            # face_crop = img[y:y2, x:x2]
            # if face_crop.size == 0:
            #     continue

            # face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            # face_crop = cv2.resize(face_crop, (160, 160))
            
            try: 
                test_embedding = get_embedding(img)

                identified_user, distance = identify_face_cosine(test_embedding, stored_embeddings, stored_labels, threshold)

                if identified_user is not None and identified_user == user: 
                    true_positive += 1
                elif identified_user is not None and identified_user != user:
                    false_positive += 1
                elif identified_user is None and user not in stored_labels: 
                    true_negative += 1
                else: 
                    false_negative += 1

                del img 

            except Exception as e:
                continue

    return true_positive, true_negative, false_positive, false_negative

def accuracy_calc(tp, tn, fp, fn):

    #precision
    if tp + fp == 0:
        precision = 0.0
    else: 
        precision = tp / (tp + fp)

    #recall
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    #f1 score
    if precision + recall == 0:
        f1_score = 0.0  
    else:
        f1_score = 2 * ((precision * recall) / (precision + recall))

    #accuracy
    if tn + fp + tp + fn == 0:
        accuracy = 0 
    else: 
        accuracy = (tn + tp)/ (tn + fp + tp + fn)

    return precision, recall, f1_score, accuracy


def precompute_embeddings():

    train_path = f'Data\\train'
    test_path = f'Data\\test'
    embeddings_path = 'Data\\embeddings.pkl'

    TEST_SIZE = 0.3
    threshold = 0.6

    preprocessing.Rescalling()
    preprocessing.PreProcess('facenet',TEST_SIZE)
    
        
    embeddings, labels = trainData(train_path)

    embeddings = embeddings.cpu().numpy()

    joblib.dump({"embeddings": embeddings, "labels": labels}, embeddings_path)


    true_positive, true_negative, false_positive, false_negative = testData(test_path, threshold)
    precision, recall, f1_score, accuracy = accuracy_calc(true_positive, true_negative, false_positive, false_negative)

    print(f"TP: {true_positive}, TN: {true_negative}, FP: {false_positive}, FN: {false_negative}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}, Accuracy: {accuracy:.3f}")

        
    precision, recall, f1_score, accuracy = accuracy_calc(true_positive, true_negative, false_positive, false_negative)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1_score]

    plt.figure(figsize=(6,4))
    plt.bar(metrics, values, color=['skyblue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    plt.show()
    

    conf_matrix = np.array([[true_positive, false_negative],
                        [false_positive, true_negative]])

    plt.figure(figsize=(5,4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'])
    plt.title("Confusion Matrix")
    plt.show()

    return (f"Saved embeddings for {len(labels)} users")
    

def load_precomputed_embeddings(embeddings_path = 'Data\\embeddings.pkl'):

    if not os.path.exists(embeddings_path):
        return None, None
    
    try:
      
        data = joblib.load(embeddings_path)

        embeddings = torch.tensor(data['embeddings'], dtype=torch.float32, device=device)        
        labels = data['labels']
        return embeddings, labels
    except Exception as e:
        return None, None

def identify_face_cosine(face_embedding, stored_embeddings, stored_labels, threshold=0.6):
    if stored_embeddings is None or len(stored_embeddings) == 0:
        return None, float('inf')

    # ensure embedding is a 1D torch tensor on same device as stored_embeddings
    if not isinstance(face_embedding, torch.Tensor):
        face_embedding = torch.tensor(face_embedding, dtype=torch.float32, device=stored_embeddings.device)
    else:
        face_embedding = face_embedding.to(stored_embeddings.device).float()

    # compute cosine similarities with all stored embeddings
    similarities = F.cosine_similarity(face_embedding.unsqueeze(0), stored_embeddings)  # shape: [num_embeddings]
    
    # find best match
    max_sim, idx = torch.max(similarities, dim=0)
    distance = 1 - max_sim.item()  # cosine distance

    if distance <= threshold:
        return stored_labels[idx.item()], distance
    else:
        return None, distance

def verificationImages(img):

    ## blurriness
    blurValue = int(cv2.Laplacian(img, cv2.CV_64F).var())
    if blurValue < 30:
        return False
        
    rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ### multiple faces
    faces = face_cascade.detectMultiScale(rgbImage, scaleFactor=1.1, minNeighbors=4)

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
        cv2.imwrite(f"Datasets\\{name}\\{name}_{count}.jpg", cv2.resize(face_crop, (250, 250)))
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
    max_count = 100 if mode == 'capture' else 100 if mode == 'attendance' else 0

    while (count < max_count if mode == 'capture' else count < max_count if mode == 'attendance' else is_updating):                
        if not is_updating:  
            break
        
        if cam is not None and cam.isOpened():
            isTrue, frame = cam.read()

            if isTrue:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_detection = face_cascade.detectMultiScale(rgbImage, scaleFactor=1.1, minNeighbors=4)       
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

                        if count >= 100:  # Stop after 100 captures
                            close_camera(cam)
                            camera_label.grid_remove()
                            is_updating = False
                            break
                        
                        cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (255, 255, 0), 2)

                    if mode == None or mode == 'attendance':
                        if(verificationImages(frame)):
                            try:
                                face_embedding = get_embedding(face_crop)

                                identified_user, distance = identify_face_cosine(face_embedding, stored_embeddings, stored_labels)
                                
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
        if(not os.path.exists("Datasets")):
            os.makedirs("Datasets")
            return True, None
    else:
        deleted, error_delete = database.deleteDatabase(database.get_connection())
            
        if deleted:
            return False, message
        else: 
            return False, error_delete
    
    return True, None





