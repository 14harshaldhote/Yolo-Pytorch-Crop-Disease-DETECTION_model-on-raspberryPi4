import cv2
import numpy as np
import pytest

from app.detector import Detector, letterbox, nms
from tests.conftest import MODEL, SAMPLE


@pytest.fixture(scope="module")
def detector():
    return Detector(MODEL, conf_threshold=0.4)


def test_model_metadata(detector):
    assert detector.names == ["good", "infected", "yellow"]
    assert detector.input_size == (640, 640)


def test_finds_infected_leaf(detector):
    detections = detector.predict(cv2.imread(str(SAMPLE)))
    assert [d.label for d in detections] == ["infected"]
    d = detections[0]
    assert d.confidence > 0.8
    h, w = cv2.imread(str(SAMPLE)).shape[:2]
    x1, y1, x2, y2 = d.box
    assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


def test_blank_image_has_no_detections(detector):
    assert detector.predict(np.full((480, 640, 3), 114, np.uint8)) == []


def test_boxes_map_back_to_original_size(detector):
    image = cv2.imread(str(SAMPLE))
    small = detector.predict(image)
    big = detector.predict(cv2.resize(image, None, fx=2, fy=2))  # same letterboxed input, 2x coordinates
    assert len(small) == len(big) == 1
    assert big[0].box == pytest.approx([v * 2 for v in small[0].box], rel=0.01)

    # Non-square input: a 640x320 image is padded by 160 px top and bottom. The same pixels
    # placed at y=160 in a 640x640 canvas give the model an identical input, so the boxes
    # must differ by exactly the padding.
    leaf = cv2.resize(image, (320, 320))
    wide = np.full((320, 640, 3), 114, np.uint8)
    wide[:, :320] = leaf
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[160:480, :320] = leaf
    a, b = detector.predict(wide), detector.predict(canvas)
    assert len(a) == len(b) == 1
    x1, y1, x2, y2 = b[0].box
    assert a[0].box == pytest.approx((x1, y1 - 160, x2, y2 - 160), abs=0.01)


def test_rejects_grayscale(detector):
    with pytest.raises(ValueError):
        detector.predict(np.zeros((64, 64), np.uint8))


def test_missing_model():
    with pytest.raises(FileNotFoundError):
        Detector("weights/does-not-exist.onnx")


def test_letterbox_pads_to_square():
    out, ratio, (px, py) = letterbox(np.zeros((480, 640, 3), np.uint8), (640, 640))
    assert out.shape == (640, 640, 3)
    assert ratio == 1 and px == 0 and py == 80


def test_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert nms(boxes, scores, 0.45) == [0, 2]


def test_parses_yolov8_layout(detector):
    """YOLOv8/YOLO11 exports put classes on axis 1 and have no objectness score."""
    nc = len(detector.names)
    pred = np.zeros((1, 4 + nc, 3), np.float32)
    pred[0, :4, 0] = [320, 320, 100, 100]
    pred[0, 4 + 1, 0] = 0.9  # infected
    pred[0, :4, 1] = [322, 322, 100, 100]
    pred[0, 4 + 1, 1] = 0.6  # overlaps box 0, same class: suppressed
    pred[0, :4, 2] = [100, 100, 40, 40]
    pred[0, 4 + 0, 2] = 0.2  # below threshold
    dets = detector.postprocess(pred, 1.0, (0, 0), (640, 640), 0.4)
    assert len(dets) == 1
    assert dets[0].label == "infected"
    assert dets[0].box == pytest.approx((270, 270, 370, 370))


def test_nms_is_per_class(detector):
    nc = len(detector.names)
    pred = np.zeros((1, 2, 5 + nc), np.float32)
    pred[0, 0, :5] = [320, 320, 100, 100, 1.0]
    pred[0, 0, 5 + 0] = 0.9  # good
    pred[0, 1, :5] = [320, 320, 100, 100, 1.0]
    pred[0, 1, 5 + 1] = 0.8  # infected, same box
    dets = detector.postprocess(pred, 1.0, (0, 0), (640, 640), 0.4)
    assert sorted(d.label for d in dets) == ["good", "infected"]
