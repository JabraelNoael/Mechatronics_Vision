from ultralytics import YOLO

best = '/opt/homebrew/runs/detect/train17/weights/best.pt'

model = YOLO("yolov8n.pt")

model.predict(source=0, show=True)