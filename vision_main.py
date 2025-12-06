import cv2
import math
import torch
import pyzed.sl as sl

# Adjust these imports to match your actual filenames / paths
from zed_wrapper import Zed              
from YoloDetection import ObjDetModel    


# based off YAML file decide, which is object 1 
TARGET_CLASS_ID = 1


def get_nearest_target_box(zed: Zed, results, min_conf: float = 0.3):
    """
    Given YOLO results and the ZED object, find the nearest detected target.

    Uses the updated Zed.get_median_distance(x1, y1, x2, y2) which returns
    Euclidean distance in meters.

    returns:
        nearest_distance: float or math.inf if none
        nearest_box: (x1, y1, x2, y2) or None
    """
    nearest_distance = math.inf
    nearest_box = None

    # To protect us from weird states, like no detection or image
    if results is None or not hasattr(results, "xyxy") or len(results.xyxy) == 0:
        return nearest_distance, nearest_box

    # tensor of shape (N, 6): [x1, y1, x2, y2, conf, cls]
    boxes = results.xyxy[0]

    # Check if tensor is empty or has 0 elements
    if boxes is None or boxes.numel() == 0:
        return nearest_distance, nearest_box

    # We loop through each detection
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        conf = float(conf)
        cls = int(cls)

        # Filter by class and confidence
        if cls != TARGET_CLASS_ID:
            continue
        if conf < min_conf:
            continue

        # Cast to ints for pixel coordinates
        x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)

        # Use your Euclidean-distance median method
        distance = zed.get_median_distance(x1_i, y1_i, x2_i, y2_i)

        # Ignore garbage distance values if they pop
        if distance <= 0 or math.isinf(distance) or math.isnan(distance):
            continue

        # Keep the nearest detection
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_box = (x1_i, y1_i, x2_i, y2_i)

    return nearest_distance, nearest_box


def draw_box_and_distance(image, box, distance):
    """
    Draws a rectangle around the target and overlays the distance in meters.
    """
    if box is None:
        return image

    x1, y1, x2, y2 = box

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Put distance text above the box
    label = f"{distance:.2f} m"
    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return image


def main():
    # ----------------- Initialize ZED camera -----------------
    zed = Zed()
    state = zed.open()
    if state != 0:  # sl.ERROR_CODE.SUCCESS has value 0
        print("Failed to open ZED camera, state:", state)
        return

    # ----------------- Initialize YOLO model -----------------
    # Uses default path inside ObjDetModel; override if needed:
    # detection = ObjDetModel(model_path="/mnt/c/Users/whift/Mechatronics_Vision/models/v1 (1).pt")
    detection = ObjDetModel()

    print("Starting main loop. Press 'q' to quit.")

    while True:
        # Get RGB image from ZED
        image = zed.get_image()
        if image is None:
            print("No image from ZED.")
            continue

        # Run YOLO detection
        results = detection.detect_in_image(image)

        # New: handle None safely (e.g., bad frame / conversion failure)
        if results is not None:
            # Find nearest target using Euclidean distance from ZED
            nearest_distance, nearest_box = get_nearest_target_box(zed, results)

            if nearest_box is not None and nearest_distance < math.inf:
                print(f"Nearest target distance: {nearest_distance:.2f} m")
                image = draw_box_and_distance(image, nearest_box, nearest_distance)
            else:
                # Optional: print if nothing detected
                # print("No valid target detected.")
                pass

        # Show image (either with box or just raw frame)
        cv2.imshow("YOLO + ZED (distance)", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
