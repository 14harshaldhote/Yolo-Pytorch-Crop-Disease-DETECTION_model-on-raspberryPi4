"""Service settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    model_path: Path = ROOT / "weights" / "best.onnx"
    conf_threshold: float = 0.4
    iou_threshold: float = 0.45
    max_detections: int = 300
    # Empty means auth is off. Set API_KEY in production.
    api_key: str = ""
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    max_concurrency: int = 2
    ort_threads: int = 0  # 0 lets ONNX Runtime pick
    rate_limit_per_minute: int = 60  # per client IP, 0 disables
    cors_origins: list[str] = field(default_factory=list)
    enable_docs: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        model_path = Path(_env_str("MODEL_PATH", str(cls.model_path)))
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        return cls(
            model_path=model_path,
            conf_threshold=_env_float("CONF_THRESHOLD", cls.conf_threshold),
            iou_threshold=_env_float("IOU_THRESHOLD", cls.iou_threshold),
            max_detections=_env_int("MAX_DETECTIONS", cls.max_detections),
            api_key=_env_str("API_KEY", ""),
            max_upload_bytes=int(_env_float("MAX_UPLOAD_MB", 10) * 1024 * 1024),
            max_image_pixels=_env_int("MAX_IMAGE_PIXELS", cls.max_image_pixels),
            max_concurrency=max(1, _env_int("MAX_CONCURRENCY", cls.max_concurrency)),
            ort_threads=_env_int("ORT_THREADS", cls.ort_threads),
            rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", cls.rate_limit_per_minute),
            cors_origins=_env_list("CORS_ORIGINS"),
            enable_docs=_env_bool("ENABLE_DOCS", cls.enable_docs),
        )
