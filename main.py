import cv2 
import mediapipe as mp
import numpy as np
mp_drawing  = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    ret, frame = cap.read()
    
    image = cv2.cvtColor(frame, cv2.COLOR_BAYER_BGR2RGB)
    image.flags.writable = False

    result = pose.process(image)
    
    image.flags.writable = True
    image = cv2.cvtColor(frame, cv2.COLOR_BAYER_RGB2BGR)
    
    cv2.imshow("", frame)    
    
    if cv2.waitKey(10) & 0xFF == ord("q"):
      break
  
cap.release()
cv2.destroyAllWindows()
