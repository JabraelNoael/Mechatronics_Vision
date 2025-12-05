import cv2
import torch
from ultralytics import YOLO

class ObjDetModel:

    """
    discord: @kialli,@Seaniiii
    github: @kchan5071,@Gabriel-Sean13

    YOLOv11 object detection wrapper.

    - Uses your own trained YOLOv11 model:
      /mnt/c/Users/whift/Mechatronics_Vision/models/v1 (1).pt

    - Keeps the old interface:
        results.xyxy[0] is a tensor of shape (N, 6):
        [x1, y1, x2, y2, conf, cls]

      so existing code such as:

          for box in results.xyxy[0]:
              if box[5] != 0:
                  continue
              ...

      still works.WIll update.
"""

    def __init__(self, model_path: str = None, device: str = None):

        print('Initializing YOLOv11 object')
     
        #Check for Available GPU
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        self.device = device

        #Model Path selection
        if model_path is None:
            # NOTE: replace this with your real model path
            model_path = "Orn path"
        
        #Load Yolo Model
        self.model = YOLO(model_path)
        #Move model to device, moves parameter to chosen device either cuda:0 or cpu
        self.model.to(self.device)
        
        #Consistency from old code from kai, This isn't used anywhere in the code base
        #Ultralytics does all the resizing work now
        self.model_resolution = 640

        print(f"YOLOv11 init success on device: {self.device}")
        print(f"model path: {model_path}")

    #This function is only for inference, no gradients.
    #This for detect_in_image()
    @torch.inference_mode()
    def detect_in_image(self,image):

        if image is None:
            return None
        
        #OpenCV images are BGR YOLO expects RGB images
        frame_cc = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #Runs YOLO and returns a list of results per image
        yolo_results = self.model(frame_cc, verbose = False)

        #Ultralytics will return an empty list this avoids that 
        if len(yolo_results) == 0:
            #If no results it create an empty .xyxy[0] to avoid crashes
            res = type("EmptyResults", (), {})()
            res.xyxy = [torch.empty((0,6), dtype=torch.float32)]
            return res
        
        #Takes the first and only image
        res = yolo_results[0]
        #Boxes Object
        boxes = res.boxes

        #To handle no detection
        if boxes is None or boxes.xyxy is None or boxes.xyxy.numel() == 0:
            res.xyxy = [torch.empty((0,6), dtype=torch.float32)]
            return res
        
        #(M,4)
        xyxy = boxes.xyxy
        #(M,1)
        conf = boxes.conf.view(-1,1)
        #(M,1)
        cls = boxes.cls.view(-1,1)

        #Combine into a (M,6) tensor
        combined = torch.cat([xyxy,conf,cls],dim = 1)

        #Attach an .xyxy to match the old YOLOv5 structure kai made
        res.xyxy = [combined]

        return res
