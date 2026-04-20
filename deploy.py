"""Live deployment for a trained YOLO model on the ZED camera (or webcam
fallback). Loads weights, runs inference per frame, overlays class name,
confidence, and (when available) per-detection distance in meters.

Usage
-----
    # ZED SDK on the ZedBox
    python deploy.py --weights runs_cnn_transfer/ft/weights/best.pt --zed

    # laptop webcam fallback (no depth)
    python deploy.py --weights runs_cnn_transfer/ft/weights/best.pt --webcam 0

    # single-image smoke test
    python deploy.py --weights runs_cnn_transfer/ft/weights/best.pt \\
                     --image data_2026/raw/ambulance_0001.png

    # record annotated video
    python deploy.py --weights best.pt --zed --record runs/live.mp4

Press `q` in the preview window to quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# Default class names pulled from vision_config.py so they stay in lockstep
# with training. Override on the CLI with --names if you trained a different map.
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from vision_config import CLASS_NAMES as DEFAULT_CLASS_NAMES
except Exception:
    DEFAULT_CLASS_NAMES = {
        0: 'firetruck',
        1: 'fire',
        2: 'ambulance',
        3: 'blood',
    }

PALETTE = [(0, 255, 0), (0, 255, 255), (255, 128, 0), (255, 0, 255),
           (0, 128, 255), (128, 255, 0)]


# ---------------------------------------------------------------------------
# Source backends
# ---------------------------------------------------------------------------
class ZedSource:
    """Thin adapter around the project's zed_wrapper.Zed — frame + depth."""

    def __init__(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from zed_wrapper import Zed
        import pyzed.sl as sl
        self._sl = sl
        self._zed = Zed()
        state = self._zed.open()
        if state != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f'ZED open failed: {state}')
        self._image_mat = sl.Mat()
        self._distance_mat = sl.Mat()

    def read(self) -> Optional[np.ndarray]:
        if self._zed.zed.grab() != self._sl.ERROR_CODE.SUCCESS:
            return None
        self._zed.zed.retrieve_image(self._image_mat, self._sl.VIEW.LEFT)
        self._zed.zed.retrieve_measure(self._distance_mat,
                                       self._sl.MEASURE.DISTANCE)
        img = np.ascontiguousarray(self._image_mat.get_data())
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def distance(self, x1: int, y1: int, x2: int, y2: int) -> Optional[float]:
        d = self._zed.get_median_distance(x1, y1, x2, y2)
        return None if d is None or d < 0 else float(d)

    def close(self) -> None:
        try:
            self._zed.zed.close()
        except Exception:
            pass


class WebcamSource:
    """cv2.VideoCapture wrapper. No depth."""

    def __init__(self, index: int) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f'could not open webcam index {index}')

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def distance(self, *a, **k) -> Optional[float]:
        return None

    def close(self) -> None:
        self._cap.release()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_detection(frame: np.ndarray, xyxy, cls_id: int, conf: float,
                   distance_m: Optional[float],
                   class_names: dict) -> None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    color = PALETTE[cls_id % len(PALETTE)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    name = class_names.get(cls_id, str(cls_id))
    tag = f'{name} {conf:.2f}'
    if distance_m is not None and distance_m > 0:
        tag += f' | {distance_m:.2f} m'
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, tag, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loops
# ---------------------------------------------------------------------------
def load_model(weights: Path, device: str):
    from ultralytics import YOLO
    if not weights.exists():
        raise FileNotFoundError(
            f'weights not found: {weights}\n'
            '→ train first (cnn_transfer.ipynb → Section A) or point --weights '
            'at a stock .pt file (e.g. yolo11n.pt).')
    model = YOLO(str(weights))
    print(f'loaded {weights}  |  device={device}')
    return model


def run_live(args, class_names: dict) -> None:
    device = args.device
    model = load_model(Path(args.weights), device)

    if args.zed:
        source = ZedSource()
        print('source: ZED SDK')
    else:
        source = WebcamSource(args.webcam)
        print(f'source: webcam {args.webcam}')

    writer: Optional[cv2.VideoWriter] = None
    window = 'deploy'
    frame_i, t0, last_annotated = 0, time.time(), None

    try:
        while True:
            frame = source.read()
            if frame is None:
                break

            # optional frame skipping
            if args.frame_skip and (frame_i % (args.frame_skip + 1) != 0) \
                    and last_annotated is not None:
                frame_i += 1
                if args.show:
                    cv2.imshow(window, last_annotated)
                if writer is not None:
                    writer.write(last_annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            results = model.predict(frame, conf=args.conf, iou=args.iou,
                                    imgsz=args.imgsz, max_det=args.max_det,
                                    device=device, verbose=False)
            r = results[0]
            annotated = frame.copy()
            n_det = 0
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                n_det = len(xyxy)
                for box, c, k in zip(xyxy, confs, cls_ids):
                    dist = source.distance(*box.astype(int)) if args.zed else None
                    draw_detection(annotated, box, int(k), float(c), dist,
                                   class_names)

            frame_i += 1
            fps = frame_i / max(time.time() - t0, 1e-6)
            cv2.putText(annotated, f'{fps:5.1f} FPS  |  {n_det} det',
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

            if args.record:
                if writer is None:
                    h, w = annotated.shape[:2]
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(args.record, fourcc, 30.0, (w, h))
                writer.write(annotated)

            last_annotated = annotated
            if args.show:
                cv2.imshow(window, annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        source.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        total = time.time() - t0
        print(f'stopped after {frame_i} frames in {total:.1f}s '
              f'({frame_i / max(total, 1e-6):.1f} FPS avg)')


def run_single_image(args, class_names: dict) -> None:
    model = load_model(Path(args.weights), args.device)
    path = Path(args.image)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    r = model.predict(img, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                      device=args.device, verbose=False)[0]
    annotated = img.copy()
    dets = []
    if r.boxes is not None:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy, confs, cls_ids):
            draw_detection(annotated, box, int(k), float(c), None, class_names)
            dets.append(dict(cls=int(k), name=class_names.get(int(k), str(k)),
                             conf=float(c),
                             xyxy=[int(v) for v in box]))
    print(f'{path.name}: {len(dets)} detection(s)')
    for d in dets:
        print(f'  {d["name"]} conf={d["conf"]:.2f} xyxy={d["xyxy"]}')

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.record), annotated)
        print('wrote', args.record)
    if args.show:
        cv2.imshow('deploy', annotated)
        print('press any key to close window...')
        cv2.waitKey(0); cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_names(spec: Optional[str]) -> dict:
    if not spec:
        return dict(DEFAULT_CLASS_NAMES)
    # format: 0:firetruck,1:fire,2:ambulance,3:blood
    out = {}
    for chunk in spec.split(','):
        k, v = chunk.split(':', 1)
        out[int(k.strip())] = v.strip()
    return out


def default_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return '0'
        if torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--weights', required=True, type=str,
                   help='path to .pt weights (e.g. runs_cnn_transfer/ft/weights/best.pt)')

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--zed',    action='store_true',  help='use ZED SDK (depth enabled)')
    src.add_argument('--webcam', type=int, metavar='IDX', help='cv2.VideoCapture index')
    src.add_argument('--image',  type=str, metavar='PATH', help='single image path')

    p.add_argument('--conf',     type=float, default=0.35)
    p.add_argument('--iou',      type=float, default=0.50)
    p.add_argument('--imgsz',    type=int,   default=640)
    p.add_argument('--max-det',  type=int,   default=50)
    p.add_argument('--frame-skip', type=int, default=0,
                   help='run inference every Nth+1 frame (0 = every frame)')
    p.add_argument('--record',   type=str, default=None,
                   help='optional output file (.mp4 for live, .png/.jpg for --image)')
    p.add_argument('--no-show',  dest='show', action='store_false')
    p.add_argument('--device',   type=str, default=default_device(),
                   help='torch device; default auto-detects cuda/mps/cpu')
    p.add_argument('--names',    type=str, default=None,
                   help='class-id → name map, e.g. "0:firetruck,1:fire,2:ambulance,3:blood"')
    args = p.parse_args()

    class_names = parse_names(args.names)
    print('class names:', class_names)

    if args.image:
        run_single_image(args, class_names)
    else:
        run_live(args, class_names)


if __name__ == '__main__':
    main()
