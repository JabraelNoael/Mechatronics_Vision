import file_integrity
from ultralytics import YOLO

file_integrity.checkDataSequencing() #If the code breaks at this line it means the contents of data/image and data/labels don't match

model = YOLO("yolov8n.yaml") #.yaml means entirely untrained model

result = model.train(data="YOLO_params.yaml",epochs=40)