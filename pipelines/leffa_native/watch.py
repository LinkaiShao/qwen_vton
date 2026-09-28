"""Local status/result collection and publication to the existing GitHub report.

Workspace-only helper; authenticated remote access comes from tools/icrn. It
publishes only this pipeline/report, under the existing shared publication lock.
"""
import argparse
import base64
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
import requests
import remote
import icrn
from report import build

CHECKOUT=Path('/mnt/nvme0/vton_region_benchmark/20260925_eight_models_v2/publish-github')
LOCK=Path('/mnt/nvme0/sku_partition_veto/20260927_v1/site_publish.lock')


def fetch(user,path,required=False):
    r=requests.get(icrn.server_url(user)+'/api/contents/'+remote.REMOTE+'/'+path,headers=icrn.H(),timeout=90)
    if r.status_code==404 and not required:return None
    r.raise_for_status();obj=r.json()
    return base64.b64decode(obj['content']) if obj.get('format')=='base64' else obj['content'].encode()


def getfile(user,path,required=False):
    data=fetch(user,path,required)
    if data is not None:
        dest=remote.LOCAL/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        return data


def collect_final(user):
    for p in ['run/RESULT.json','run/DISPLAY_CASES.json','run/GRADIENT_VERIFICATION.json','run/TIMESTEP_VERIFICATION.json']:
        getfile(user,p,True)
    keys=json.loads((remote.LOCAL/'run/DISPLAY_CASES.json').read_text())['keys']
    for arm in ['initial','diffusion','correspondence','dino_all','dino_low']:
        getfile(user,'run/samples/'+arm+'/RESULT.json',True)
        if arm!='initial':getfile(user,'run/'+arm+'/COMPLETE.json',True)
        if arm in ['initial','dino_all']:getfile(user,'run/causal/'+arm+'/RESULT.json',True)
        if arm=='initial':continue
        for key in keys:
            getfile(user,f'run/samples/{arm}/{key}.png',True)
            if arm=='diffusion':getfile(user,f'run/samples/{arm}/{key}_errors.npz',True)
            if arm in ['diffusion','dino_all']:
                for suffix in ['.png','.npz']:getfile(user,f'run/traces/{arm}/{key}{suffix}',True)


def publish():
    build(remote.LOCAL)
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        def git(*args):return subprocess.run(['git',*args],cwd=CHECKOUT,check=True,capture_output=True,text=True).stdout.strip()
        if git('diff','--cached','--name-only'):raise RuntimeError('Another change is staged; defer publication')
        git('pull','--ff-only')
        code=CHECKOUT/'pipelines/leffa_native';code.mkdir(parents=True,exist_ok=True)
        for path in remote.HERE.iterdir():
            if path.suffix in ['.py','.md','.txt']:shutil.copy2(path,code/path.name)
        site=CHECKOUT/'docs/leffa-native';site.mkdir(parents=True,exist_ok=True)
        shutil.copytree(remote.LOCAL/'site',site,dirs_exist_ok=True)
        git('add','pipelines/leffa_native','docs/leffa-native')
        if git('diff','--cached','--name-only'):
            git('commit','-m','Add native LeFFA attention training and current experiment report')
            git('push','origin','master')
        sha=git('rev-parse','HEAD')
        (remote.LOCAL/'PUBLISHED.json').write_text(json.dumps({'commit':sha,'url':'https://linkaishao.github.io/qwen_vton/leffa-native/','time':time.time()}))
        print('PUBLISHED',sha,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--no-publish',action='store_true');args=p.parse_args()
    last=None;deadline=time.time()+24*3600
    while time.time()<deadline:
        try:
            user=icrn.me()['name'];raw=getfile(user,'STATE.json')
            if raw is None:time.sleep(30);continue
            state=json.loads(raw)
            for path in ['run/PROGRESS.json','run/GRADIENT_VERIFICATION.json','run/TIMESTEP_VERIFICATION.json']:
                getfile(user,path)
            current=state['stage']
            if current=='completed':collect_final(user)
            if current!=last:
                if args.no_publish:build(remote.LOCAL)
                else:publish()
                last=current
            print('WATCH',current,flush=True)
            if args.once or current in ['completed','failed']:return
        except Exception as exc:
            print('WATCH_RETRY',type(exc).__name__,str(exc)[:300],flush=True)
            if args.once:raise
        time.sleep(60)


if __name__=='__main__':main()
