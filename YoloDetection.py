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

      still works. Will update.
    """

    def __init__(self, model_path: str = None, device: str = None) -> None:
        # Do NOT load the model here – keep init light for Orin boot.
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_path = model_path or "Orn path"  
        #lazy-loaded on first use
        self.model = None  
        #Not used kai old code had this
        self.model_resolution = 640  

        print(f"ObjDetModel created. Device={self.device}, model_path={self.model_path}")
        print("YOLO model will be loaded lazily on first detect_in_image() call.")

    def _ensure_model_loaded(self) -> None:
        """
        Lazily load the YOLO model the first time we need it.
        Safe to call multiple times – it will only load once.
        """
        if self.model is not None:
            return

        print("Initializing YOLOv11 model (lazy load)...")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        print(f"YOLOv11 init success on device: {self.device}")
        print(f"Model path: {self.model_path}")

    # This function is only for inference, no gradients.
    # This is for detect_in_image()
    @torch.inference_mode()
    def detect_in_image(self, image):
        # Basic sanity checks so cv2 doesn't explode
        if image is None or not hasattr(image, "shape") or getattr(image, "size", 0) == 0:
            return None

        # Ensure the model is loaded (lazy load)
        self._ensure_model_loaded()

        # OpenCV images are BGR; YOLO expects RGB images
        frame_cc = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Runs YOLO and returns a list of results per image
        yolo_results = self.model(frame_cc, verbose=False)

        # Ultralytics will return an empty list; this avoids that causing issues
        if len(yolo_results) == 0:
            # If no results, create an empty .xyxy[0] to avoid crashes
            res = type("EmptyResults", (), {})()
            res.xyxy = [torch.empty((0, 6), dtype=torch.float32, device=self.device)]
            return res

        # Takes the first and only image
        res = yolo_results[0]
        # Boxes object
        boxes = res.boxes

        # To handle no detection gracefully
        if boxes is None or boxes.xyxy is None or boxes.xyxy.numel() == 0:
            res.xyxy = [torch.empty((0, 6), dtype=torch.float32, device=self.device)]
            return res

        # (M, 4)
        xyxy = boxes.xyxy
        # (M, 1)
        conf = boxes.conf.view(-1, 1)
        # (M, 1)
        cls = boxes.cls.view(-1, 1)

        # Combine into a (M, 6) tensor
        combined = torch.cat([xyxy, conf, cls], dim=1)

        # Attach an .xyxy to match the old YOLOv5 structure Kai made
        res.xyxy = [combined]

        return res
