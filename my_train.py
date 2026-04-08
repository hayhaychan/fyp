from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model


if __name__ == "__main__":
    
    model = YOLO("yolov8n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="mydataset.yaml", epochs=100, imgsz=640)


    