"""Live detection from a camera on the Raspberry Pi, with no PyTorch and no server.

python scripts/camera_detect.py --source 0            # USB webcam
python scripts/camera_detect.py --source 0 --show     # with a preview window
python scripts/camera_detect.py --source leaf.mp4 --save out.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.detector import Detector, draw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(ROOT / "weights" / "best.onnx"))
    parser.add_argument("--source", default="0", help="camera index, video file or stream URL")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--width", type=int, default=640, help="camera capture width")
    parser.add_argument("--height", type=int, default=480, help="camera capture height")
    parser.add_argument("--show", action="store_true", help="show a preview window (needs a desktop)")
    parser.add_argument("--save", help="write the annotated video to this .mp4 file")
    args = parser.parse_args()

    detector = Detector(args.model, conf_threshold=args.conf)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Could not open source {args.source!r}. Try `ls /dev/video*` to find the camera index.")

    writer = None
    frames, start = 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector.predict(frame)
            frames += 1
            fps = frames / (time.perf_counter() - start)
            counts = Counter(d.label for d in detections)
            summary = ", ".join(f"{n} {label}" for label, n in sorted(counts.items())) or "nothing"
            print(f"frame {frames}: {summary}  ({fps:.1f} FPS)", flush=True)

            if args.show or args.save:
                annotated = draw(frame, detections)
                if args.save:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                    writer.write(annotated)
                if args.show:
                    try:
                        cv2.imshow("Spinach leaf disease detection (q to quit)", annotated)
                    except cv2.error:
                        sys.exit("--show needs the GUI build of OpenCV: pip install opencv-python")
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
