"""FastAPI service for spinach leaf disease detection.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.detector import Detector, draw
from app.security import RateLimiter, api_key_header, check_api_key

logger = logging.getLogger("crop_disease_api")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
CHUNK = 1024 * 1024


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        detector = Detector(
            settings.model_path,
            conf_threshold=settings.conf_threshold,
            iou_threshold=settings.iou_threshold,
            max_detections=settings.max_detections,
            threads=settings.ort_threads,
        )
        detector.warmup()  # first run is slow; pay it at startup, not on a request
        app.state.detector = detector
        app.state.semaphore = asyncio.Semaphore(settings.max_concurrency)
        if not settings.api_key:
            logger.warning("API_KEY is not set: the detection endpoints are open to anyone who can reach them")
        logger.info("Loaded %s (classes: %s)", detector.model_name, ", ".join(detector.names))
        yield

    app = FastAPI(
        title="Spinach Leaf Disease Detection API",
        version="2.0.0",
        description="Detects good, infected and yellow spinach leaves with a YOLO model on ONNX Runtime.",
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
    app.state.settings = settings
    limiter = RateLimiter(settings.rate_limit_per_minute)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "Content-Type"],
        )

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # Reject oversized bodies before they are read. The streaming check in
        # read_upload covers clients that omit Content-Length.
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_upload_bytes + 64 * 1024:
            return JSONResponse({"detail": "Upload too large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "Internal server error"}, status_code=500)  # no stack traces to clients

    def authorize(request: Request, key: str | None = Security(api_key_header)) -> None:
        limiter.check(request)  # count failed attempts too, so keys can't be brute-forced
        check_api_key(settings.api_key, key)

    async def read_upload(file: UploadFile) -> np.ndarray:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, f"Unsupported type; send one of {sorted(ALLOWED_TYPES)}")
        data = bytearray()
        while chunk := await file.read(CHUNK):
            data += chunk
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, "Upload too large")
        if not data:
            raise HTTPException(422, "Empty file")
        # Read only the header first: checks the real format (not the client's claim)
        # and the pixel count before decoding, which stops decompression bombs.
        try:
            with Image.open(io.BytesIO(data)) as probe:
                fmt, (w, h) = probe.format, probe.size
        except (OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(422, "File is not a valid image") from None
        if fmt not in ALLOWED_FORMATS:
            raise HTTPException(415, f"Unsupported image format {fmt}")
        if w * h > settings.max_image_pixels:
            raise HTTPException(413, "Image has too many pixels")
        buf = np.frombuffer(bytes(data), dtype=np.uint8)
        image = await run_in_threadpool(cv2.imdecode, buf, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(422, "File is not a valid image")
        return image

    @app.get("/health", tags=["service"])
    async def health(request: Request):
        ready = getattr(request.app.state, "detector", None) is not None
        return {"status": "ok" if ready else "loading"}

    @app.get("/v1/model", tags=["service"], dependencies=[Depends(authorize)])
    async def model_info(request: Request):
        d: Detector = request.app.state.detector
        return {
            "model": d.model_name,
            "classes": d.names,
            "input_size": list(d.input_size),
            "conf_threshold": d.conf_threshold,
            "iou_threshold": d.iou_threshold,
        }

    @app.post("/v1/detect", tags=["detection"], dependencies=[Depends(authorize)])
    async def detect(
        request: Request,
        file: UploadFile = File(..., description="Leaf photo (JPEG, PNG, WebP or BMP)"),
        conf: float | None = Query(None, ge=0.01, le=1.0, description="Confidence threshold override"),
        annotate: bool = Query(False, description="Return the image with boxes drawn instead of JSON"),
    ):
        image = await read_upload(file)
        detector: Detector = request.app.state.detector
        async with request.app.state.semaphore:
            start = time.perf_counter()
            detections = await run_in_threadpool(detector.predict, image, conf)
            inference_ms = (time.perf_counter() - start) * 1000

        counts = {name: 0 for name in detector.names}
        for d in detections:
            counts[d.label] += 1

        if annotate:
            ok, jpg = await run_in_threadpool(
                cv2.imencode, ".jpg", draw(image, detections), [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not ok:
                raise HTTPException(500, "Could not encode image")
            return Response(jpg.tobytes(), media_type="image/jpeg", headers={"X-Detection-Counts": json.dumps(counts)})

        h, w = image.shape[:2]
        return {
            "model": detector.model_name,
            "image": {"width": w, "height": h},
            "inference_ms": round(inference_ms, 1),
            "counts": counts,
            "diseased": counts.get("infected", 0) + counts.get("yellow", 0) > 0,
            "detections": [d.to_dict() for d in detections],
        }

    return app


app = create_app()
