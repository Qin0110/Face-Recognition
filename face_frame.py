import cv2 as cv
import os
import tkinter as tk
import easygui as eg
import matplotlib

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml') #cascade pre-trained classifier for face detection
capture = cv.VideoCapture(0)
currentFrame = 0 

input_name = eg.enterbox("Enter your name:", "Face Frame Capture")
if input_name:
    input_name = input_name.strip()  
else:
    print("No name entered")
    exit()

if not os.path.exists(f'captured_frames/{input_name}'):
    os.makedirs(f'captured_frames/{input_name}')

while True:
    ret, frame = capture.read()
    faceData = None
    
    if not ret:
        continue # Skip the rest of the loop if frame not captured

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayFrame, 1.1, 4)
    
    for(x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faceData =  frame[y:y+h, x:x+w] # Extracting the face region
        
    cv.imshow('Capture', frame)
    
    if faceData is not None:
        cv.imwrite(f'captured_frames/{input_name}/frame_{currentFrame}.jpg', faceData)
        print("Face Detected")
    else:
        print("No face detected in this frame.")
    currentFrame += 1

    # break the loop - stop the capture if 'q' is pressed [customize sendiri]
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()