"""Exercise real weights over HTTP using native CPU model processes; no Docker needed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def main():
    processes=[]
    with tempfile.TemporaryDirectory() as tmp:
        logs=[]
        try:
            for role,port in [('seg',18761),('pose',18762)]:
                log=open(Path(tmp)/(role+'.log'),'w+');logs.append(log)
                env=dict(os.environ,MODEL_PATH=str(ROOT/'models'/('animal_pose_v2.pt' if role=='pose' else 'animal_seg.pt')),MODEL_DEVICE='cpu',YOLO_AUTOINSTALL='false',PYTHONPATH=str(ROOT))
                processes.append(subprocess.Popen([sys.executable,'-m','uvicorn','metrology.model_service:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=env,stdout=log,stderr=log))
            deadline=time.monotonic()+90
            for port in [18761,18762]:
                while True:
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:
                            assert json.load(response)['ready']
                        break
                    except Exception:
                        if time.monotonic()>deadline or any(p.poll() is not None for p in processes):
                            raise RuntimeError('Model startup failed')
                        time.sleep(.25)
            os.environ.update(SEG_SERVICE_URL='http://127.0.0.1:18761',POSE_SERVICE_URL='http://127.0.0.1:18762')
            sys.path.insert(0,str(ROOT))
            from metrology.pipeline import run
            result=run(ROOT/'examples/images/000000005992.jpg',Path(tmp)/'result',device='cpu')
            assert result['animals']
            assert (Path(tmp)/'result/measurements.csv').is_file()
            print(json.dumps({'test':'real_cpu_http_inference','animals':len(result['animals']),
                              'pose_matches':sum(a['pose'] is not None for a in result['animals']),
                              'runtime':result['runtime']},indent=2))
        except Exception:
            for log in logs:
                log.flush();log.seek(0);print(log.read())
            raise
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            for log in logs:log.close()

if __name__=='__main__':main()
