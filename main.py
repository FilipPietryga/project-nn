import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

cap = 0

def capture():
  print("inside capture()")
  cap = cv2.VideoCapture(0)
  while cap.isOpened():
    returned, frame = cap.read()
    cv2.imshow('OpenCV feed', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break
    
def postCapture():
  cap.release()
  cv2.destroyAllWindows()
  

capture()
postCapture()