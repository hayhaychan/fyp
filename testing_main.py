import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import cvzone

model = YOLO("/Users/hayhay/Documents/ultralytics-main/runs/detect/train11/weights/best.pt")

results = model.predict(source=0, show=True)

print (results)

if cv2.waitKey(1) == ord('q'):
    cv2.destroyAllWindows()