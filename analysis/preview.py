import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Paths
img_dir = r"D:\Mechatronics_Vision\data\images\train"
ann_dir = r"D:\Mechatronics_Vision\data\labels\train"

model = YOLO(r"D:\Mechatronics_Vision\keep\best.pt")

# Specify the image filenames you want to visualize (without extension or with)
selected_images = [
    "screenshot_0000.png",
    "screenshot_0001.png",
    "screenshot_0002.png",
    "screenshot_0004.png",
    "screenshot_0008.png",
    "screenshot_0023.png",
    "screenshot_0029.png",
    "screenshot_0034.png",
    "screenshot_5087.png",
    "screenshot_5089.png"
]
def view_ground_truth():
    for img_name in selected_images:
        img_path = os.path.join(img_dir, img_name)
        txt_name = img_name.replace(".png", ".txt")  # matching label name
        txt_path = os.path.join(ann_dir, txt_name)

        # Read image
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # Read annotation
        with open(txt_path, "r") as f:
            lines = f.readlines()

        # Draw each bounding box
        for line in lines:
            cls, x, y, bw, bh = map(float, line.strip().split())
            cls = int(cls)

            # Convert normalized coordinates to pixel values
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            # Draw rectangle and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, 'target', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Ground truth: {img_name}")
        plt.axis("off")
        plt.show()
def view_yolo_inference():
    for img_name in selected_images:
        img_path = os.path.join(img_dir, img_name)

        # Run inference (returns Results object)
        results = model(img_path, conf=0.25)  # you can adjust confidence threshold if needed

        # Get first result
        res = results[0]

        # Get bounding boxes, class ids, and confidences
        boxes = res.boxes.xyxy.cpu().numpy()      # [x1, y1, x2, y2]
        confs = res.boxes.conf.cpu().numpy()      # confidence scores
        clss = res.boxes.cls.cpu().numpy().astype(int)  # class ids

        # Read original image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw YOLO predictions in RED
        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[cls]} {conf:.2f}" if hasattr(model, 'names') else f"cls {cls} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show image
        plt.imshow(img)
        plt.title(f"Predicted: {img_name}")
        plt.axis("off")
        plt.show()

def view_comparison():
    for img_name in selected_images:
        img_path = os.path.join(img_dir, img_name)
        txt_name = img_name.replace(".png", ".txt")
        txt_path = os.path.join(ann_dir, txt_name)

        # ========== Ground Truth ==========
        gt_img = cv2.imread(img_path)
        h, w, _ = gt_img.shape
        with open(txt_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            cls, x, y, bw, bh = map(float, line.strip().split())
            cls = int(cls)
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(gt_img, 'target', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        # ========== YOLO Prediction ==========
        results = model(img_path, conf=0.25)
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        pred_img = cv2.imread(img_path)
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[cls]} {conf:.2f}" if hasattr(model, 'names') else f"class_{cls} {conf:.2f}"
            cv2.rectangle(pred_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(pred_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # ========== Display Side-by-Side ==========
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(gt_img)
        axes[0].set_title("Ground Truth (Green)")
        axes[0].axis("off")

        axes[1].imshow(pred_img)
        axes[1].set_title("YOLO Prediction (Red)")
        axes[1].axis("off")

        fig.suptitle(f"Comparison: {img_name}", fontsize=14)
        plt.tight_layout()
        plt.show()


view_comparison()