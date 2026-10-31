import io
import tempfile
import unittest
import json
import time
import threading
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from metrology.web import create_app


class WebTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app(output_dir=Path(self.tmp.name))
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None,None,None)
        self.tmp.cleanup()

    def test_page_and_samples(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        samples = self.client.get('/api/samples').json()
        self.assertGreaterEqual(len(samples), 1)
        self.assertEqual(self.client.get(samples[0]['url']).status_code, 200)

    def test_bad_image_rejected(self):
        response = self.client.post('/api/jobs', files={'image': ('bad.jpg', b'bad', 'image/jpeg')})
        self.assertEqual(response.status_code, 422)

    def test_unknown_sample_and_job(self):
        self.assertEqual(self.client.post('/api/jobs', data={'sample_id': '../secret'}).status_code, 422)
        self.assertEqual(self.client.get('/api/jobs/missing').status_code, 404)
        self.assertEqual(self.client.get('/api/jobs/missing/artifacts/result.json').status_code, 404)

    def test_artifact_not_arbitrary_file(self):
        self.assertEqual(self.client.get('/api/jobs/123/artifacts/.env').status_code, 404)

    def test_threshold_validation(self):
        self.assertEqual(self.client.post('/api/jobs', data={'sample_id':'1818','eye_conf':1.5}).status_code,422)

    def test_upload_limit(self):
        response = self.client.post('/api/jobs', files={'image': ('large.jpg', b'x'*(10*1024*1024+1), 'image/jpeg')})
        self.assertEqual(response.status_code,413)

    def test_work_lifecycle_and_downloads(self):
        # Replace only expensive model inference; HTTP, input normalization,
        # job storage, busy guard and artifact restrictions all run for real.
        started, release = threading.Event(), threading.Event()
        def inference(image, out, **options):
            started.set()
            release.wait(5)
            out.mkdir()
            (out/'result.json').write_text('{"animals": []}')
            return {'animals': []}
        with patch('metrology.web.run', side_effect=inference):
            response = self.client.post('/api/jobs', data={'sample_id':'1818'})
            self.assertEqual(response.status_code,202)
            job_id = response.json()['id']
            self.assertTrue(started.wait(5))
            self.assertEqual(self.client.get(f'/api/jobs/{job_id}').json()['status'],'running')
            self.assertEqual(self.client.post('/api/jobs', data={'sample_id':'1818'}).status_code,409)
            self.assertEqual(self.client.get(f'/api/jobs/{job_id}/artifacts/result.json').status_code,404)
            release.set()
            deadline = time.monotonic()+5
            while time.monotonic()<deadline:
                job = self.client.get(f'/api/jobs/{job_id}').json()
                if job['status']=='complete':
                    break
                time.sleep(.01)
            self.assertEqual(job['status'],'complete')
            self.assertEqual(self.client.get(f'/api/jobs/{job_id}/artifacts/result.json').json(),{'animals':[]})
            self.assertEqual(self.client.get(f'/api/jobs/{job_id}/artifacts/source.jpg').status_code,200)
            self.assertEqual(self.client.get(f'/api/jobs/{job_id}/artifacts/.env').status_code,404)

    def test_inference_failure_is_visible(self):
        with patch('metrology.web.run', side_effect=RuntimeError('device test failure')):
            response = self.client.post('/api/jobs', data={'sample_id':'1818'})
            job_id=response.json()['id']
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                job=self.client.get(f'/api/jobs/{job_id}').json()
                if job['status']=='failed':
                    break
                time.sleep(.01)
            self.assertEqual(job['status'],'failed')
            self.assertIn('device test failure',job['error'])
