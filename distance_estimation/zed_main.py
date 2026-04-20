import sys
import numpy as np
import pyzed.sl as sl
import cv2
import logging
import getopt

from ultralytics import YOLO

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(argv):
    # CLI defaults
    yolo_weights_path = r"/Users/seaniiii/Code_Stuff/Mechatronics_Vision/models/v1.pt"  # your trained YOLOv8 model
    svo_path = None
    zed_id = 0

    # Configure and check back 
    help_str = (
        "zed_yolov8_zdepth.py "
        "-w <best.pt> "
        "[-s <svo_file>] "
        "[-z <zed_id>]"
    )

    try:
        opts, args = getopt.getopt(argv, "hw:s:z:", ["weight=", "svo_file=", "zed_id="])
    except getopt.GetoptError:
        log.exception(help_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            log.info(help_str)
            sys.exit()
        elif opt in ("-w", "--weight"):
            yolo_weights_path = arg
        elif opt in ("-s", "--svo_file"):
            svo_path = arg
        elif opt in ("-z", "--zed_id"):
            zed_id = int(arg)

    # ZED input type
    input_type = sl.InputType()
    if svo_path is not None:
        log.info("SVO file: %s", svo_path)
        input_type.set_from_svo_file(svo_path)
    else:
        input_type.set_from_camera_id(zed_id)

    # Open ZED 
    zed = sl.Camera()
    init = sl.InitParameters(input_t=input_type)
    
    #Change Resolution here when needed
    init.camera_resolution = sl.RESOLUTION.HD1080

    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    # In Meters
    init.coordinate_units = sl.UNIT.METER 

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        sys.exit(1)

    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Mats 
    image_size = zed.get_camera_information().camera_resolution
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()  # XYZRGBA in meters

    # YOLOv8
    log.info("Loading YOLOv8 model: %s", yolo_weights_path)
    yolo_model = YOLO(yolo_weights_path)

    # Names from model
    if isinstance(yolo_model.names, dict):
        LABELS = [yolo_model.names[i] for i in range(len(yolo_model.names))]
    else:
        LABELS = list(yolo_model.names)

    # Debug overlay
    COLOR_BOX = (0, 0, 255)
    COLOR_TEXT = (0, 255, 255)

    CONF_THRES = 0.50
    IOU_THRES = 0.40


    # Most recent output only
    # last_z_depth is the outwards pass
    last_z_depth_m = None     
    
    #Not needed but just in case
    last_class_name = None     
    last_conf = None           

    exit_flag = True
    while exit_flag:
        err = zed.grab(runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

        frame_rgba = image_zed.get_data()

        #Modify to pass specific images, doulbe check once code gets texted to work 
        img = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

        # reset each frame
        last_z_depth_m = None
        last_class_name = None
        last_conf = None

        results = yolo_model.predict(source=img, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
        r0 = results[0]

        if r0.boxes is not None and len(r0.boxes) > 0:

            # Since only one class/object will appear at a time
            # just take the highest-confidence detection.
            box = max(r0.boxes, key=lambda b: float(b.conf[0].cpu().numpy()))

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            left, top, right, bottom = int(x1), int(y1), int(x2), int(y2)

            # center pixel for Z-depth lookup
            cx = int((left + right) / 2)
            cy = int((top + bottom) / 2)
            cx = max(0, min(cx, img.shape[1] - 1))
            cy = max(0, min(cy, img.shape[0] - 1))

            # Z depth (meters)
            _, pc_val = point_cloud.get_value(cx, cy)
            Z = pc_val[2]
            if np.isfinite(Z):
                last_z_depth_m = float(Z)

            last_class_name = LABELS[cls_id] if 0 <= cls_id < len(LABELS) else str(cls_id)
            last_conf = conf

            # Optional debug draw
            cv2.rectangle(img, (left, top), (right, bottom), COLOR_BOX, 2)
            cv2.circle(img, (cx, cy), 4, COLOR_BOX, -1)
            cv2.putText(
                img,
                f"{last_class_name}: {last_conf:.2f}",
                (left, max(0, top - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_BOX,
                2,
            )
            cv2.putText(
                img,
                f"Z: {last_z_depth_m:.2f} m" if last_z_depth_m is not None else "Z: --",
                (left, top + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_TEXT,
                2,
            )

        # last_z_depth_m is updated every frame and holds ONLY the most recent Z-depth in meters.
        # Pass last_z_depth_m to other classes as needed.

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit_flag = False

    cv2.destroyAllWindows()
    zed.close()
    print("\nFINISH")


if __name__ == "__main__":
    main(sys.argv[1:])