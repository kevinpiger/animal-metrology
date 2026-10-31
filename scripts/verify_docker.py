"""Smoke test the public Web entry point of an already-running Compose deployment."""
import argparse
import json
import time
import urllib.request
import uuid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8080')
    args=parser.parse_args()
    base=args.url.rstrip('/')
    def request(path,data=None,headers=None):
        with urllib.request.urlopen(urllib.request.Request(base+path,data=data,headers=headers or {}),timeout=30) as response:
            return response.read()
    health=json.loads(request('/api/health'))
    assert health['ready'],health
    page=request('/').decode()
    assert 'id="removeUpload"' in page and 'id="settingsDialog"' in page
    assert '/static/metrics.html' not in page and 'API 文件' not in page
    spec=json.loads(request('/openapi.json'))
    assert '/api/jobs' in spec['paths']
    assert b'swagger' in request('/docs').lower()
    samples=json.loads(request('/api/samples'))
    assert len(samples)==3
    boundary='metrology-'+uuid.uuid4().hex
    fields={'sample_id':'5992','device':'cpu','eye_conf':'0.5'}
    body=''.join(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n' for k,v in fields.items())+f'--{boundary}--\r\n'
    job=json.loads(request('/api/jobs',body.encode(),{'Content-Type':'multipart/form-data; boundary='+boundary}))['id']
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        result=json.loads(request('/api/jobs/'+job))
        if result['status']=='failed':raise RuntimeError(result['error'])
        if result['status']=='complete':break
        time.sleep(.5)
    else:raise TimeoutError('Analysis timed out')
    assert result['result']['animals']
    for file in ['source.jpg','overlay.jpg','result.json','measurements.csv','animal_0_mask.png']:
        assert request('/api/jobs/'+job+'/artifacts/'+file)
    print(json.dumps({'health':health,'job_id':job,'animals':len(result['result']['animals']),
                      'pose_matches':sum(a['pose'] is not None for a in result['result']['animals']),
                      'all_downloads':True,'swagger':True,'current_ui':True},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
