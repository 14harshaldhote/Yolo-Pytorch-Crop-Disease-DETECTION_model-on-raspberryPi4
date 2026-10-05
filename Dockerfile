# Detection API image. Builds for x86-64 and 64-bit ARM (Raspberry Pi 4):
#   docker build -t crop-disease-api .
#   docker run -p 8000:8000 -e API_KEY=change-me crop-disease-api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY app ./app
COPY weights/best.onnx ./weights/best.onnx

# Never run as root inside the container.
RUN useradd --system --uid 10001 --no-create-home api
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-server-header"]
