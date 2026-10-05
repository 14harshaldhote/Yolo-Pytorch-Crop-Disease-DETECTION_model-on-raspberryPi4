import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from tests.conftest import MODEL

KEY = "test-key-123"


def make_client(**overrides) -> TestClient:
    settings = Settings(model_path=MODEL, api_key=KEY, rate_limit_per_minute=0, **overrides)
    return TestClient(create_app(settings))


@pytest.fixture(scope="module")
def client():
    with make_client() as c:
        yield c


def upload(content: bytes, name="leaf.jpg", ctype="image/jpeg"):
    return {"file": (name, content, ctype)}


def auth():
    return {"X-API-Key": KEY}


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert r.headers["x-content-type-options"] == "nosniff"


def test_model_info(client):
    r = client.get("/v1/model", headers=auth())
    assert r.status_code == 200
    assert r.json()["classes"] == ["good", "infected", "yellow"]


def test_detect(client, sample_bytes):
    r = client.post("/v1/detect", files=upload(sample_bytes), headers=auth())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["counts"] == {"good": 0, "infected": 1, "yellow": 0}
    assert body["diseased"] is True
    assert body["detections"][0]["label"] == "infected"
    assert body["image"] == {"width": 640, "height": 640}


def test_detect_conf_override(client, sample_bytes):
    r = client.post("/v1/detect?conf=0.99", files=upload(sample_bytes), headers=auth())
    assert r.status_code == 200 and r.json()["detections"] == []


def test_detect_annotated(client, sample_bytes):
    r = client.post("/v1/detect?annotate=true", files=upload(sample_bytes), headers=auth())
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert json.loads(r.headers["x-detection-counts"])["infected"] == 1
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR) is not None


def test_png_upload(client):
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (114, 114, 114)).save(buf, format="PNG")
    r = client.post("/v1/detect", files=upload(buf.getvalue(), "x.png", "image/png"), headers=auth())
    assert r.status_code == 200 and r.json()["detections"] == []


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_requires_api_key(client, sample_bytes, headers):
    assert client.post("/v1/detect", files=upload(sample_bytes), headers=headers).status_code == 401
    assert client.get("/v1/model", headers=headers).status_code == 401


def test_rejects_wrong_content_type(client, sample_bytes):
    r = client.post("/v1/detect", files=upload(sample_bytes, "a.txt", "text/plain"), headers=auth())
    assert r.status_code == 415


def test_rejects_non_image_bytes(client):
    r = client.post("/v1/detect", files=upload(b"not really a jpeg"), headers=auth())
    assert r.status_code == 422


def test_rejects_disguised_format(client):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="GIF")
    r = client.post("/v1/detect", files=upload(buf.getvalue()), headers=auth())
    assert r.status_code == 415


def test_rejects_empty_file(client):
    assert client.post("/v1/detect", files=upload(b""), headers=auth()).status_code == 422


def test_rejects_bad_conf(client, sample_bytes):
    r = client.post("/v1/detect?conf=5", files=upload(sample_bytes), headers=auth())
    assert r.status_code == 422


def test_rejects_large_upload(sample_bytes):
    with make_client(max_upload_bytes=10_000) as c:
        r = c.post("/v1/detect", files=upload(sample_bytes), headers=auth())
        assert r.status_code == 413


def test_rejects_too_many_pixels():
    buf = io.BytesIO()
    Image.new("RGB", (3000, 3000)).save(buf, format="PNG")
    with make_client(max_image_pixels=4_000_000) as c:
        r = c.post("/v1/detect", files=upload(buf.getvalue(), "big.png", "image/png"), headers=auth())
        assert r.status_code == 413


def test_rate_limit(sample_bytes):
    settings = Settings(model_path=MODEL, api_key=KEY, rate_limit_per_minute=2)
    with TestClient(create_app(settings)) as c:
        codes = [c.get("/v1/model", headers=auth()).status_code for _ in range(3)]
        assert codes == [200, 200, 429]


def test_rate_limit_counts_failed_auth():
    settings = Settings(model_path=MODEL, api_key=KEY, rate_limit_per_minute=2)
    with TestClient(create_app(settings)) as c:
        codes = [c.get("/v1/model", headers={"X-API-Key": "guess"}).status_code for _ in range(3)]
        assert codes == [401, 401, 429]


def test_open_mode_without_key(sample_bytes):
    with TestClient(create_app(Settings(model_path=MODEL, rate_limit_per_minute=0))) as c:
        assert c.post("/v1/detect", files=upload(sample_bytes)).status_code == 200


def test_docs_can_be_disabled():
    with make_client(enable_docs=False) as c:
        assert c.get("/docs").status_code == 404
        assert c.get("/openapi.json").status_code == 404
