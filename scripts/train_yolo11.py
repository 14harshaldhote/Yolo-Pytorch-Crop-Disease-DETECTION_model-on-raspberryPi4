"""Retrain the detector with Ultralytics YOLO11 and export it for the API.

Run on a machine with a GPU (Google Colab works). Needs `pip install ultralytics roboflow`
and the dataset in ./vf-2 (see README). The API and camera script accept the exported
ONNX file as is: set MODEL_PATH=weights/yolo11.onnx.

    python scripts/train_yolo11.py --model yolo11s.pt --epochs 300
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11s.pt", help="yolo11n.pt is faster on a Pi, yolo11s.pt more accurate")
    parser.add_argument("--data", default=str(ROOT / "data" / "spinach.yaml"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=100)
    args = parser.parse_args()
    os.chdir(ROOT)  # data/spinach.yaml points at ./vf-2

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(ROOT / "runs" / "train"),
        name="yolo11",
    )
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print(f"mAP@0.5 = {metrics.box.map50:.3f}, mAP@0.5:0.95 = {metrics.box.map:.3f}")

    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True))
    target = ROOT / "weights" / "yolo11.onnx"
    shutil.copy(onnx_path, target)
    print(f"Exported {target}. Compare its mAP with the YOLOv5 model (0.702) before switching.")


if __name__ == "__main__":
    main()
