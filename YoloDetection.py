import cv2
import torch
from ultralytics import YOLO

"""
    discord: @kialli
    github: @kchan5071

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

      still works.
"""


class ObjDetModel:

    def __init__(self, model_path: str = None, device: str = None):
        print("initializing YOLOv11 object")

        # ------------------ device selection ------------------
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        self.device = device

        # ------------------ model path ------------------
        # Default to your trained model
        if model_path is None:
            model_path = "/mnt/c/Users/whift/Mechatronics_Vision/models/v1 (1).pt"

        # ------------------ load YOLOv11 model ------------------
        self.model = YOLO(model_path)
        # Move underlying model to device (YOLO returns a wrapper object)
        self.model.to(self.device)

        # Keep this for consistency with old code; YOLO will internally handle resizing,
        # but we can leave this here in case you want to reintroduce manual resizing later.
        self.model_resolution = 640

        print(f"YOLOv11 init success on device: {self.device}")
        print(f"model path: {model_path}")

    def load_new_model(self, model_path: str):
        """
            Load a new YOLOv11 model from a given path.
        """
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"loaded new YOLOv11 model: {model_path}")

    #This function is only for inference; optimize it, no gradients.
    @torch.inference_mode()
    ################################################################
    
    def detect_in_image(self, image):
        """
            Run YOLOv11 detection on a BGR image (OpenCV format).

            input:
                image: np_array (H, W, 3) in BGR

            return:
                results: a YOLO Results object (for this single image)
                         with an added attribute .xyxy such that:
                             results.xyxy[0] -> (N, 6) tensor:
                                 [x1, y1, x2, y2, conf, cls]
                         in ORIGINAL image coordinates.
        """
        if image is None:
            return None

        # Convert BGR -> RGB for YOLO
        frame_cc = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run YOLOv11; returns a list of Results (one per image)
        yolo_results = self.model(frame_cc, verbose=False)

        if len(yolo_results) == 0:
            # No results at all: create an empty .xyxy[0] to avoid crashes
            res = type("EmptyResults", (), {})()
            res.xyxy = [torch.empty((0, 6), dtype=torch.float32)]
            return res

        # First (and only) image
        res = yolo_results[0]
        boxes = res.boxes  # Boxes object

        # If no detections, create empty tensor with same structure
        if boxes is None or boxes.xyxy is None or boxes.xyxy.numel() == 0:
            res.xyxy = [torch.empty((0, 6), dtype=torch.float32)]
            return res

        # YOLOv11 gives:
        #   boxes.xyxy : (N, 4) -> [x1, y1, x2, y2]
        #   boxes.conf : (N,)    confidences
        #   boxes.cls  : (N,)    class indices
        xyxy = boxes.xyxy              # (N, 4)
        conf = boxes.conf.view(-1, 1)  # (N, 1)
        cls  = boxes.cls.view(-1, 1)   # (N, 1)

        # Combine into (N, 6): [x1, y1, x2, y2, conf, cls]
        combined = torch.cat([xyxy, conf, cls], dim=1)

        # Attach an .xyxy attribute matching the old YOLOv5-hub structure
        # so existing code can still do results.xyxy[0]
        res.xyxy = [combined]

        return res
