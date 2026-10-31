FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR=/tmp/ultralytics
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir "torch==2.15.0" "torchvision>=0.21,<1" --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r requirements.txt
# OpenCV 5 ARM wheel uses unsupported instructions on some virtual CPUs.
RUN pip install --no-cache-dir "opencv-python==4.11.0.86"
COPY metrology ./metrology
RUN useradd --uid 10001 --create-home app
USER app
CMD ["sh", "-c", "exec uvicorn metrology.model_service:app --host 0.0.0.0 --port ${SERVICE_PORT:-8000}"]
