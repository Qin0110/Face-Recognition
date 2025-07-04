import cv2 as cv
import os
import tkinter as tk
import easygui as eg

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
    if not ret:
        continue

    cv.imshow('Capture', frame)
    cv.imwrite(f'captured_frames/{input_name}/frame_{currentFrame}.jpg', frame)
    currentFrame += 1

    # break the loop - stop the capture if 'q' is pressed [customize sendiri]
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()