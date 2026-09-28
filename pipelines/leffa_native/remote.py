"""Resumable upload and owned H200 job, using the existing private ICRN token."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import requests
from concurrent.futures import ThreadPoolExecutor

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'tools/icrn'))
import icrn

REMOTE='leffa_native/20260928_v1'
LOCAL=Path('/mnt/nvme0/leffa_native/20260928_v1')


def put(user,path,data):
    r=requests.put(icrn.server_url(user)+'/api/contents/'+path,headers=icrn.H(),
                   json={'type':'file','format':'base64','content':base64.b64encode(data).decode()},timeout=180)
    r.raise_for_status()


def execute(user,code,timeout=120):
    rc=icrn.execute(user,code,timeout=timeout)
    if rc:raise RuntimeError('Remote action failed: '+str(rc))


def upload():
    u=icrn.me();assert icrn.running(u);user=u['name']
    execute(user,f'from pathlib import Path\np=Path.home()/{REMOTE!r}\np.mkdir(parents=True,exist_ok=True)\n(p/"code").mkdir(exist_ok=True)\n(p/"upload").mkdir(exist_ok=True)')
    for source in HERE.iterdir():
        if source.suffix in ['.py','.md','.txt']:put(user,REMOTE+'/code/'+source.name,source.read_bytes())
    put(user,REMOTE+'/manifest.json',(LOCAL/'manifest.json').read_bytes())
    records=json.loads((LOCAL/'manifest.json').read_text())['records']
    state=LOCAL/'UPLOADED.json'
    uploaded=set(json.loads(state.read_text())['keys']) if state.exists() else set()
    waiting_since=time.time()
    while len(uploaded)<len(records):
        ready=[r for r in records if r['key'] not in uploaded and (LOCAL/'data'/r['key']/'READY.json').exists()][:128]
        if not ready:
            if time.time()-waiting_since>7200:raise TimeoutError('Local data preparation stalled for two hours')
            time.sleep(5);continue
        waiting_since=time.time()
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w') as tar:
            for row in ready:tar.add(LOCAL/'data'/row['key'],arcname='data/'+row['key'])
        blob=buffer.getvalue();sha=hashlib.sha256(blob).hexdigest();prefix='batch_'+sha[:16]
        parts=[]
        uploads=[]
        for i in range(0,len(blob),4<<20):
            name=prefix+f'.{i//(4<<20):03d}';parts.append(name)
            uploads.append((REMOTE+'/upload/'+name,blob[i:i+(4<<20)]))
        # Independent immutable chunks; extraction still waits for every part.
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(put,user,path,data) for path,data in uploads]
            for future in futures:future.result()
        code=f'''from pathlib import Path
import tarfile,hashlib
p=Path.home()/{REMOTE!r};archive=p/'upload'/{(prefix+'.tar')!r}
with archive.open('wb') as f:
    for name in {parts!r}:f.write((p/'upload'/name).read_bytes())
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sha!r}
with tarfile.open(archive) as tar:tar.extractall(p,filter='data')
archive.unlink()
for name in {parts!r}:(p/'upload'/name).unlink()
print('BATCH_READY', {len(ready)!r})
'''
        execute(user,code)
        uploaded.update(r['key'] for r in ready)
        state.write_text(json.dumps({'keys':sorted(uploaded),'remote':REMOTE,'updated':time.time()}))
        print('UPLOADED',len(uploaded),len(records),flush=True)
    while not (LOCAL/'DATA_READY.json').exists():time.sleep(2)
    put(user,REMOTE+'/DATA_READY.json',(LOCAL/'DATA_READY.json').read_bytes())
    print('ALL_DATA_UPLOADED',flush=True)


def upload_code():
    user=icrn.me()['name']
    for source in HERE.iterdir():
        if source.suffix in ['.py','.md','.txt']:put(user,REMOTE+'/code/'+source.name,source.read_bytes())


def run():
    u=icrn.me();assert icrn.running(u)
    code=f'''import subprocess,time,json,os,signal
from pathlib import Path
p=Path.home()/{REMOTE!r};log=p/'controller.log'
python=Path.home()/'vton_region_benchmark/20260924_eight_models/job/envs/leffa/bin/python'
env=dict(os.environ,PYTHONUNBUFFERED='1',HF_HOME=str(Path.home()/'sgn/hf'),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',PYTHONPATH=str(p/'deps'))
with log.open('a') as f:
    proc=subprocess.Popen([str(python),'-u',str(p/'code/supervise.py'),'--root',str(p)],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    (p/'controller.json').write_text(json.dumps({{'pid':proc.pid,'started':time.time(),'command':'supervise.py'}}))
    print('SUPERVISOR_PID',proc.pid,flush=True)
    last=''
    try:
        while proc.poll() is None:
            time.sleep(15)
            tail=log.read_text()[-1500:]
            if tail!=last:print(tail,flush=True);last=tail
        (p/'EXIT.json').write_text(json.dumps({{'returncode':proc.returncode,'finished':time.time()}}))
        print('SUPERVISOR_EXIT',proc.returncode,flush=True)
    except BaseException:
        os.killpg(proc.pid,signal.SIGTERM)
        raise
'''
    return icrn.execute(u['name'],code,timeout=24*3600)


def status():
    user=icrn.me()['name']
    for path in ['STATE.json','run/PROGRESS.json','run/GRADIENT_VERIFICATION.json','run/RESULT.json','EXIT.json']:
        r=requests.get(icrn.server_url(user)+'/api/contents/'+REMOTE+'/'+path,headers=icrn.H(),timeout=30)
        if r.status_code==404:continue
        r.raise_for_status();obj=r.json()
        data=base64.b64decode(obj['content']).decode() if obj.get('format')=='base64' else obj['content']
        print(path,data[:4000])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['upload','upload-code','run','status'])
    args=p.parse_args()
    if args.action=='upload':upload()
    elif args.action=='upload-code':upload_code()
    elif args.action=='status':status()
    else:raise SystemExit(run())
