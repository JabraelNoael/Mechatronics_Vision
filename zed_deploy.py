"""Simple YOLO + ZED inference with depth readouts.
Tuned for weak hardware (low resolution, fast depth mode).
SDK 5.x compatible.

Edit WEIGHTS below, then run:
    python3 zed_deploy.py
Press 'q' in the preview window to quit.
"""
import sys
import numpy as np
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

# ─── EDIT THESE ─────────────────────────────────────────────────────────
WEIGHTS     = 'runs_cnn_transfer/v1.1/weights/best.pt'
CONF_THRES  = 0.7
IOU_THRES   = 0.50
RESOLUTION  = sl.RESOLUTION.HD1080 # VGA | HD720 | HD1080 | HD2K
FPS         = 10
IMGSZ       = 640
MAX_DET     = 4 #Value limits how many inferences can be made at once e.g. 4 means only show top 4 or less most confident predictors per frame
CLASSES     = None #None allows for all classes, use a list of IDs for only the classes you want e.g. [0,3] only shows 0 and 3
AUGMENT     = False
# ────────────────────────────────────────────────────────────────────────

# Load YOLO
print(f'loading {WEIGHTS}...')
model = YOLO(WEIGHTS)
print(f'classes: {model.names}')

# Open ZED
zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = RESOLUTION
init.camera_fps = FPS
init.depth_mode = sl.DEPTH_MODE.PERFORMANCE   # fastest depth mode
init.coordinate_units = sl.UNIT.METER

err = zed.open(init)
if err != sl.ERROR_CODE.SUCCESS:
    print(f'ZED open failed: {err}')
    sys.exit(1)
print('ZED opened')

# SDK 5.x: camera info is nested in camera_configuration
cam_info = zed.get_camera_information()
res = cam_info.camera_configuration.resolution
print(f'stream resolution: {res.width}x{res.height}')

runtime = sl.RuntimeParameters()
image_mat = sl.Mat()
point_cloud = sl.Mat()

DEPTH_COLOR = (0, 255, 255)

print('running — press q to quit')
try:
    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        frame = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_RGBA2BGR)

        results = model.predict(frame, conf=CONF_THRES, iou=IOU_THRES, imgsz=IMGSZ, max_det=MAX_DET, classes=CLASSES, augment=AUGMENT, verbose=False)
        r = results[0]
        annotated = r.plot()

        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx = max(0, min((x1 + x2) // 2, frame.shape[1] - 1))
                cy = max(0, min((y1 + y2) // 2, frame.shape[0] - 1))
                _, pc_val = point_cloud.get_value(cx, cy)
                z = pc_val[2]
                txt = f'{z:.2f} m' if np.isfinite(z) else '-- m'
                cv2.putText(annotated, txt, (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, DEPTH_COLOR, 2)
                cv2.circle(annotated, (cx, cy), 4, (255, 255, 255), -1)

        cv2.imshow('ZED + YOLO (press q)', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    zed.close()
    cv2.destroyAllWindows()
    print('done')
