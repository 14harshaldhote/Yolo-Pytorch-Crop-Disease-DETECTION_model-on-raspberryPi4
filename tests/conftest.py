from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "weights" / "best.onnx"
SAMPLE = ROOT / "runs" / "detect" / "exp" / "DSC_1362_JPG.rf.8546a40f637e8916b154ecefefb542f1.jpg"


@pytest.fixture(scope="session")
def sample_bytes() -> bytes:
    return SAMPLE.read_bytes()
