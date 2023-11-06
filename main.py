import colorsys
import numpy as np
import pyvirtualcam
import cv2
from deepface import DeepFace # Used to detect emotion

with pyvirtualcam.Camera(width=512, height=512, fps=30, print_fps=True) as cam:
    print(f'Using virtual camera: {cam.device}')
    frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    while True:
        try:
            # Get frame from actual webcam
            vc = cv2.VideoCapture(0)
            _, frame = vc.read()
            vc.release()
            # Save frame to file
            cv2.imwrite('frame.png', frame)
            # Detect emotion
            emotion = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            print(emotion) 
            img = "cam_"
            # Get emotion
            emotion = emotion[0]['dominant_emotion']
            img += emotion + ".png"
            frame = cv2.imread(img, 4)
            # White background
            frame[np.all(frame == [0, 0, 0], axis=-1)] = [255, 255, 255]
            cam.send(frame)
            cam.sleep_until_next_frame()
        except KeyboardInterrupt: # Exit on Ctrl+C
            break
        except:
            continue