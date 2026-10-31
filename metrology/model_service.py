"""One process and one weight per internal model service."""
import base64
import os
import threading
from contextlib import asynccontextmanager
from io import BytesIO
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from PIL import Image, ImageOps
from .pipeline import load_model
from ultralytics.utils.patches import _image_open as PIL_OPEN
from .runtime import select_device

state = {}
lock = threading.Lock()

@asynccontextmanager
async def lifespan(app):
    runtime = select_device(os.getenv('MODEL_DEVICE', 'cpu'))
    model = load_model(os.environ['MODEL_PATH'])
    model.predict(np.zeros((64,64,3), dtype=np.uint8), device=runtime['device'], verbose=False)
    state.update(model=model, runtime=runtime)
    yield
    state.clear()

app = FastAPI(lifespan=lifespan)

@app.get('/health')
def health():
    return {'ready': bool(state), 'runtime': state.get('runtime')}

@app.post('/predict')
def predict(image: UploadFile = File(...), device: str = Form('auto', pattern='^(auto|cuda|mps|cpu)$'),
            conf: float = Form(.25, ge=0, le=1)):
    raw = image.file.read(64*1024*1024+1)
    if len(raw) > 64*1024*1024:
        raise HTTPException(413, 'Image too large')
    try:
        with PIL_OPEN(BytesIO(raw), formats=['JPEG', 'PNG', 'WEBP']) as im:
            if im.width*im.height > 20_000_000:
                raise ValueError('Too many pixels')
            img = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(im).convert('RGB')), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise HTTPException(422, 'Invalid image') from exc
    configured = os.getenv('MODEL_DEVICE', 'cpu')
    requested = 'cpu' if configured == 'cpu' else configured if device == 'auto' else device
    with lock:
        runtime = select_device(requested)
        pred = state['model'].predict(img, conf=conf, device=runtime['device'], retina_masks=True, verbose=False)[0]
    masks=[]
    if pred.masks is not None:
        for m in pred.masks.data.cpu().numpy():
            mask=cv2.resize(m, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_NEAREST)
            ok, png=cv2.imencode('.png', (mask>.5).astype('uint8')*255)
            if not ok:
                raise RuntimeError('Mask encoding failed')
            masks.append(base64.b64encode(png).decode('ascii'))
    return {'names': pred.names, 'boxes': pred.boxes.data.cpu().tolist() if pred.boxes is not None else [],
            'keypoints': pred.keypoints.data.cpu().tolist() if pred.keypoints is not None else [],
            'masks': masks, 'runtime': runtime}
