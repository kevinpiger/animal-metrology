import base64
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from fastapi.testclient import TestClient
from metrology.pipeline import run
from metrology.remote import predict_remote
from metrology.web import create_app


class ServiceTests(unittest.TestCase):
    def payload(self, pose=False):
        _, png = cv2.imencode('.png', np.ones((12,12), dtype=np.uint8)*255)
        return {'names': {'0':'dog'}, 'boxes': [[0,0,12,12,.9,0]],
                'masks': [] if pose else [base64.b64encode(png).decode()],
                'keypoints': [[[3,4,.9],[6,8,.9]]+[[11,11,.9]]*15] if pose else [],
                'runtime': {'device':'cpu','requested_device':'auto','fallback_reasons':[]}}

    def test_remote_pipeline_preserves_masks_and_measurement(self):
        class Response:
            def __init__(self, payload): self.payload=payload
            def raise_for_status(self): pass
            def json(self): return self.payload
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); image=root/'image.png'
            cv2.imwrite(str(image),np.zeros((12,12,3),dtype=np.uint8))
            with patch.dict(os.environ, {'SEG_SERVICE_URL':'http://seg:8000','POSE_SERVICE_URL':'http://pose:8000'}), \
                 patch('httpx.Client.post', side_effect=lambda url,**kw: Response(self.payload(pose='pose' in url))):
                result=run(image,root/'result')
            self.assertEqual(result['animals'][0]['inter_eye_distance_px'],5.)
            self.assertEqual(result['runtime']['services']['pose']['device'],'cpu')
            self.assertTrue((root/'result/animal_0_mask.png').is_file())

    def test_remote_empty_detections(self):
        from types import SimpleNamespace
        payload={'names':{'0':'dog'},'boxes':[],'masks':[],'keypoints':[], 'runtime':{'device':'cpu'}}
        response=SimpleNamespace(raise_for_status=lambda:None,json=lambda:payload)
        with patch('httpx.Client.post',return_value=response):
            result,_=predict_remote('http://seg',np.zeros((12,12,3),dtype=np.uint8),'auto',.25)
        self.assertEqual(len(result.boxes),0)
        self.assertIsNone(result.masks)

    def test_expiry_removes_state_and_files(self):
        def inference(image,out,**kw):
            out.mkdir(); (out/'result.json').write_text('{}'); return {}
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'RESULT_TTL_SECONDS':'1'}):
            with TestClient(create_app(d)) as client, patch('metrology.web.run',side_effect=inference):
                job=client.post('/api/jobs',data={'sample_id':'1818'}).json()['id']
                deadline=time.monotonic()+3
                while time.monotonic()<deadline:
                    if client.get('/api/jobs/'+job).json()['status']=='complete': break
                    time.sleep(.01)
                time.sleep(1.1)
                self.assertEqual(client.get('/api/jobs/'+job).status_code,404)
                self.assertFalse((Path(d)/job).exists())

    def test_model_endpoint_serializes_prediction(self):
        from metrology import model_service
        from ultralytics.engine.results import Results
        import torch
        image=np.zeros((12,12,3),dtype=np.uint8)
        pred=Results(orig_img=image,path='',names={0:'dog'},boxes=torch.tensor([[0,0,12,12,.9,0]]),masks=torch.ones((1,12,12)))
        class Model:
            def predict(self,*args,**kw): return [pred]
        with patch.dict(os.environ, {'MODEL_PATH':'unused.pt','MODEL_DEVICE':'cpu'}), patch.object(model_service,'load_model',return_value=Model()):
            with TestClient(model_service.app) as client:
                self.assertTrue(client.get('/health').json()['ready'])
                _, png=cv2.imencode('.png',image)
                response=client.post('/predict',files={'image':('image.png',png.tobytes(),'image/png')},data={'device':'cuda'})
                self.assertEqual(response.status_code,200)
                self.assertEqual(len(response.json()['masks']),1)
                self.assertEqual(response.json()['runtime']['device'],'cpu')
                self.assertEqual(client.post('/predict',files={'image':('x',b'bad')}).status_code,422)
