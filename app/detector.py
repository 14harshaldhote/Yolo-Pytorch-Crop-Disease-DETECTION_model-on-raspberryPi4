"""ONNX Runtime detector for YOLO models.

Runs without PyTorch, which keeps it small and fast on a Raspberry Pi. Handles both
output layouts:
  * YOLOv5:        (1, N, 5 + nc)  rows of cx, cy, w, h, objectness, class scores
  * YOLOv8 / 11:   (1, 4 + nc, N)  columns of cx, cy, w, h, class scores
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "box": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
        }


def letterbox(image: np.ndarray, size: tuple[int, int], color: int = 114):
    """Resize keeping aspect ratio and pad to `size` (h, w). Returns image, ratio, (pad_x, pad_y)."""
    h, w = image.shape[:2]
    ratio = min(size[0] / h, size[1] / w)
    new_w, new_h = round(w * ratio), round(h * ratio)
    if (new_w, new_h) != (w, h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size[1] - new_w) / 2, (size[0] - new_h) / 2
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return image, ratio, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Plain non-maximum suppression on xyxy boxes. Returns kept indices, best first."""
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def _parse_names(raw: str | None, num_classes: int) -> list[str]:
    if raw:
        # literal_eval, never eval: the metadata comes from a file.
        names = ast.literal_eval(raw)
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        return [str(n) for n in names]
    return [f"class{i}" for i in range(num_classes)]


class Detector:
    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        threads: int = 0,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.model_name = model_path.name

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        self.input_size = (h if isinstance(h, int) else 640, w if isinstance(w, int) else 640)

        out_shape = self.session.get_outputs()[0].shape
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = _parse_names(meta.get("names"), self._guess_num_classes(out_shape))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    @staticmethod
    def _guess_num_classes(shape) -> int:
        dims = [d for d in shape[1:] if isinstance(d, int)]
        return max(min(dims) - 5, 1) if dims else 1

    def warmup(self) -> None:
        self.predict(np.zeros((*self.input_size, 3), dtype=np.uint8))

    def preprocess(self, image_bgr: np.ndarray):
        padded, ratio, pad = letterbox(image_bgr, self.input_size)
        blob = cv2.dnn.blobFromImage(padded, scalefactor=1 / 255.0, swapRB=True)  # NCHW float32 RGB
        return blob, ratio, pad

    def postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        pad: tuple[int, int],
        image_shape: tuple[int, int],
        conf_threshold: float,
    ) -> list[Detection]:
        pred = output[0]
        nc = len(self.names)
        if pred.shape[0] == 4 + nc and pred.shape[1] != 4 + nc:
            pred = pred.T  # YOLOv8/11: (4 + nc, N) -> (N, 4 + nc)
            boxes, class_scores = pred[:, :4], pred[:, 4:]
        elif pred.shape[1] == 5 + nc:
            boxes, class_scores = pred[:, :4], pred[:, 5:] * pred[:, 4:5]  # YOLOv5 objectness
        else:
            raise ValueError(f"Unsupported model output shape {output.shape} for {nc} classes")

        class_ids = class_scores.argmax(1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]
        mask = scores > conf_threshold
        if not mask.any():
            return []
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

        xyxy = np.empty_like(boxes)
        xyxy[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
        xyxy[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2

        # Offset boxes by class so one NMS pass never suppresses across classes.
        offset = class_ids[:, None].astype(np.float32) * 4096.0
        keep = nms(xyxy + offset, scores, self.iou_threshold)[: self.max_detections]

        xyxy = xyxy[keep]
        xyxy[:, [0, 2]] -= pad[0]
        xyxy[:, [1, 3]] -= pad[1]
        xyxy /= ratio
        h, w = image_shape
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)

        return [
            Detection(int(c), self.names[int(c)], float(s), tuple(float(v) for v in b))
            for b, s, c in zip(xyxy, scores[keep], class_ids[keep], strict=True)
        ]

    def predict(self, image_bgr: np.ndarray, conf_threshold: float | None = None) -> list[Detection]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image of shape (H, W, 3)")
        blob, ratio, pad = self.preprocess(image_bgr)
        output = self.session.run(None, {self.input_name: blob})[0]
        conf = self.conf_threshold if conf_threshold is None else conf_threshold
        return self.postprocess(output, ratio, pad, image_bgr.shape[:2], conf)


COLORS = {"good": (60, 180, 75), "infected": (40, 40, 220), "yellow": (0, 200, 230)}


def draw(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of the image with boxes and labels drawn."""
    out = image_bgr.copy()
    thickness = max(2, round(sum(out.shape[:2]) / 600))
    for d in detections:
        color = COLORS.get(d.label, (255, 128, 0))
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = f"{d.label} {d.confidence:.2f}"
        scale = thickness / 3
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thickness - 1))
        y_text = max(y1, th + 4)
        cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 2, y_text), color, -1)
        cv2.putText(
            out,
            text,
            (x1 + 1, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return out
