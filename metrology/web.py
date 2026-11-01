"""Loopback-only validation workbench with bounded background inference."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
import json
import re
import threading
import uuid
import os
import time
import shutil
import httpx

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from .pipeline import run
# Pinned Ultralytics patches Image.open globally. Use its saved Pillow decoder
# for our explicit JPEG/PNG/WebP allowlist, independently of module import order.
from ultralytics.utils.patches import _image_open as PIL_OPEN
from .runtime import select_device

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    {'id':'1818','name':'斑馬・遮擋案例','file':'000000001818.jpg','animal_count':2},
    {'id':'5992','name':'羊群・多實例','file':'000000005992.jpg','animal_count':5},
    {'id':'7281','name':'馬匹・雙動物','file':'000000007281.jpg','animal_count':2},
]
MAX_BYTES = int(os.getenv('MAX_UPLOAD_MB', '10')) * 1024 * 1024
MAX_PIXELS = 20_000_000


def create_app(output_dir=None):
    output = Path(output_dir) if output_dir is not None else Path(os.getenv('TEMP_DIR', str(ROOT/'outputs/web')))
    jobs, lock = {}, threading.Lock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='measurement')
    runtime_cache = {}

    ttl = max(1, int(os.getenv('RESULT_TTL_SECONDS', '1800')))
    stop = threading.Event()

    def cleanup():
        while not stop.wait(min(30, ttl)):
            with lock:
                expired = [k for k,v in jobs.items() if v.get('_finished_at') is not None and time.monotonic()-v['_finished_at'] >= ttl]
                for k in expired:
                    jobs.pop(k)
                    shutil.rmtree(output/k, ignore_errors=True)

    @asynccontextmanager
    async def lifespan(app):
        cleaner = threading.Thread(target=cleanup, daemon=True)
        cleaner.start()
        try:
            yield
        finally:
            executor.shutdown(wait=True)
            stop.set()
            cleaner.join()
            with lock:
                for k in jobs:
                    shutil.rmtree(output/k, ignore_errors=True)
                jobs.clear()

    app = FastAPI(title='Animal Metrology — 本地驗證', version='0.1.0', lifespan=lifespan)
    app.mount('/static', StaticFiles(directory=ROOT/'static'), name='static')
    app.mount('/samples', StaticFiles(directory=ROOT/'examples/images'), name='samples')

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(ROOT/'static/index.html')

    @app.get('/api/samples')
    def samples():
        return [dict(s, url='/samples/'+s['file'], source_url='http://images.cocodataset.org/val2017/'+s['file']) for s in SAMPLES]

    @app.get('/api/health')
    def health():
        if os.getenv('SEG_SERVICE_URL') and os.getenv('POSE_SERVICE_URL'):
            services = {}
            for name in ('seg', 'pose'):
                try:
                    with httpx.Client(timeout=5, trust_env=False) as client:
                        response = client.get(os.environ[name.upper()+'_SERVICE_URL']+'/health')
                        response.raise_for_status()
                        services[name] = response.json()
                except Exception:
                    services[name] = {'ready': False}
            return {'ready': all(v.get('ready',False) for v in services.values()), 'services': services,
                    'runtime': {'device': services['seg'].get('runtime',{}).get('device','unavailable')},
                    'models': {'animal_seg.pt':services['seg'].get('ready',False),'animal_pose_v2.pt':services['pose'].get('ready',False)},
                    'scope':'temporary_only','max_upload_mb':10}
        models = {name:(ROOT/'models'/name).is_file() for name in ('animal_seg.pt','animal_pose_v2.pt')}
        if not runtime_cache:
            try:
                runtime_cache.update(select_device())
            except RuntimeError as exc:
                return {'ready':False, 'models':models, 'error':str(exc)}
        return {'ready':all(models.values()), 'models':models, 'runtime':runtime_cache,
                'scope':'local_only', 'max_upload_mb':10}

    def execute(job_id, image, options):
        with lock:
            jobs[job_id]['status'] = 'running'
        try:
            result = run(image, output/job_id/'result', **options)
            # Serve a concise stable schema; weights remain local.
            with lock:
                jobs[job_id].update(status='complete', result=result, _finished_at=time.monotonic())
        except Exception as exc:
            with lock:
                jobs[job_id].update(status='failed', error=f'{type(exc).__name__}: {exc}', _finished_at=time.monotonic())

    @app.post('/api/jobs', status_code=202)
    def create_job(sample_id: str | None = Form(None), image: UploadFile | None = File(None),
                   eye_conf: float = Form(float(os.getenv('EYE_CONF', '.5')), ge=0, le=1),
                   conf: float = Form(float(os.getenv('DETECTION_CONF', '.25')), ge=0, le=1),
                   match_iou: float = Form(float(os.getenv('MATCH_IOU', '.3')), gt=0, le=1),
                   device: str = Form(os.getenv('MODEL_DEVICE', 'cpu'), pattern='^(auto|mps|cuda|cpu)$')):
        if bool(sample_id) == bool(image):
            raise HTTPException(422, '請選擇一張範例或上傳一張圖片。')
        if image:
            content = image.file.read(MAX_BYTES+1)
            if len(content)>MAX_BYTES:
                raise HTTPException(413, '圖片不可超過 10 MB。')
        else:
            sample = next((s for s in SAMPLES if s['id']==sample_id),None)
            if sample is None:
                raise HTTPException(422, '找不到這個範例。')
            content = (ROOT/'examples/images'/sample['file']).read_bytes()
        try:
            with PIL_OPEN(BytesIO(content), formats=['JPEG', 'PNG', 'WEBP']) as uploaded:
                if uploaded.width * uploaded.height > MAX_PIXELS:
                    raise HTTPException(413, '圖片不可超過 2,000 萬像素。')
                normalized = ImageOps.exif_transpose(uploaded).convert('RGB')
                normalized.load()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(422, '無法解碼圖片，請選擇有效的 JPG 或 PNG。')
        with lock:
            if any(job['status'] in ('queued','running') for job in jobs.values()):
                raise HTTPException(409, '已有分析進行中，請等待完成再試。')
            job_id = uuid.uuid4().hex
            folder = output/job_id
            folder.mkdir(parents=True)
            source = folder/'source.jpg'
            normalized.save(source, quality=95)
            jobs[job_id] = {'id':job_id,'status':'queued','sample_id':sample_id}
            executor.submit(execute, job_id, source,
                            {'device':device,'conf':conf,'eye_conf':eye_conf,'match_iou':match_iou})
        return {'id':job_id,'status':'queued'}

    def get_job(job_id):
        with lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(404, '找不到分析工作；服務重啟後請重新分析。')
            if job.get('_finished_at') is not None and time.monotonic()-job['_finished_at'] >= ttl:
                jobs.pop(job_id)
                shutil.rmtree(output/job_id, ignore_errors=True)
                raise HTTPException(404, 'Result expired')
            return {k:v for k,v in job.items() if not k.startswith('_')}

    @app.get('/api/jobs/{job_id}')
    def job_status(job_id: str):
        return get_job(job_id)

    @app.get('/api/jobs/{job_id}/artifacts/{name}')
    def artifact(job_id: str, name: str):
        job = get_job(job_id)
        if name == 'source.jpg':
            path = output/job_id/name
        else:
            if job['status']!='complete':
                raise HTTPException(404,'分析完成後才可下載結果。')
            if name not in {'overlay.jpg','result.json','measurements.csv'} and not re.fullmatch(r'animal_\d+_mask\.png',name):
                raise HTTPException(404,'找不到這個產物。')
            path = output/job_id/'result'/name
        if not path.is_file():
            raise HTTPException(404,'找不到這個產物。')
        return FileResponse(path)

    return app


app = create_app()
