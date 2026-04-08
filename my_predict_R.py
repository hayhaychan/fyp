from ultralytics import YOLO

# Use 0 for built-in camera, 1 or 2 for external
model = YOLO("/Users/hayhay/Documents/ultralytics-main/runs/detect/train11/weights/best.pt")
results = model.predict(source=0, show=True)
print(results)
