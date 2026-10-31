import base64
from types import SimpleNamespace
import cv2
import numpy as np
import httpx
import torch
from ultralytics.engine.results import Results


def predict_remote(url, image, device, conf):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('Cannot encode input image')
    with httpx.Client(timeout=180, trust_env=False) as client:
        response = client.post(url + '/predict', files={'image': ('image.png', encoded.tobytes(), 'image/png')},
                               data={'device': device, 'conf': conf})
        response.raise_for_status()
        data = response.json()
    masks = [cv2.imdecode(np.frombuffer(base64.b64decode(m), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 0
             for m in data['masks']]
    result = Results(orig_img=image, path='', names={int(k): v for k,v in data['names'].items()},
                     boxes=torch.tensor(data['boxes'], dtype=torch.float32).reshape(-1,6),
                     masks=torch.from_numpy(np.asarray(masks, dtype=np.float32)) if masks else None,
                     keypoints=torch.tensor(data['keypoints'], dtype=torch.float32) if data['keypoints'] else None)
    return result, data['runtime']
