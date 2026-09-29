"""Bounded, resumable localization audit. Owns only its subprocess groups; never trains."""
import fcntl
import signal
import subprocess
import traceback
from common import *

LEFFA='/mnt/nvme0/leffa_native/env/bin/python'
TEACHER='/home/link/venvs/ootd/bin/python'

def run(root,max_hours=9,publishing=True):
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    lock=(root/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bind_gpu();os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';os.environ['LEFFA_WORKSPACE']=str(WORKSPACE)
    os.environ['PYTHONUNBUFFERED']='1'
    begin=time.time();deadline=begin+max_hours*3600;child=None;last_publish=0
    code_hashes={p.name:digest(p) for p in HERE.glob('*.py')}
    atomic(root/'RUN.json',{'pid':os.getpid(),'started':begin,'deadline':deadline,'gpu':GPU,
                          'code_sha256':code_hashes,'training_enabled':False,'DINO_loaded':False})
    stages=[('reference_names',TEACHER,'annotate.py',['names']),
            ('correct_masks_and_details',TEACHER,'masks.py',[]),
            ('frozen_model_smoke',LEFFA,'probe.py',['smoke','--limit','2']),
            ('calibration_probe',LEFFA,'probe.py',['probe','--role','calibration']),
            ('freeze_readout',LEFFA,'probe.py',['calibrate']),
            ('regression_probe',LEFFA,'probe.py',['probe','--role','regression']),
            ('mask_comparison',LEFFA,'probe.py',['compare-masks']),
            ('heldout_probe',LEFFA,'probe.py',['probe','--role','heldout']),
            ('all_timestep_sweep',LEFFA,'probe.py',['sweep']),
            ('causal_interventions',LEFFA,'probe.py',['causal']),
            ('generated_image_annotations',TEACHER,'annotate.py',['generated']),
            ('evaluate_gate',LEFFA,'evaluate.py',[])]
    def stop(signum,frame):raise KeyboardInterrupt('Supervisor stop requested')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def publish_now():
        nonlocal last_publish
        if not publishing:return
        log=root/'logs/publication.log'
        with log.open('a') as out:
            try:
                rc=subprocess.run([LEFFA,str(HERE/'publish.py'),'--root',str(root)],stdout=out,stderr=subprocess.STDOUT,timeout=90).returncode
                atomic(root/'PUBLICATION_STATUS.json',{'returncode':rc,'time':time.time()})
            except Exception as exc:atomic(root/'PUBLICATION_STATUS.json',{'error':str(exc),'time':time.time()})
        last_publish=time.time()
    try:
        for stage,python,script,args in stages:
            if time.time()>=deadline:raise TimeoutError('Audit time budget reached; no training is started')
            if any(digest(HERE/k)!=v for k,v in code_hashes.items()):raise RuntimeError('Audit code changed during execution; resume explicitly after review')
            done=root/'stages'/(stage+'.json')
            if done.exists() and read(done).get('code_sha256')==code_hashes:continue
            state(root,stage,supervisor_pid=os.getpid(),deadline=deadline)
            publish_now()
            command=[python,str(HERE/script),'--root',str(root),*args]
            with (root/'logs'/(stage+'.log')).open('a') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic(root/'WORKER.json',{'pid':child.pid,'stage':stage,'command':command,'started':time.time()})
                while child.poll() is None:
                    if time.time()>=deadline:raise TimeoutError('Audit time budget reached; worker stopped')
                    if time.time()-last_publish>=600:publish_now()
                    time.sleep(5)
                if child.returncode:raise RuntimeError(f'{stage} failed (exit {child.returncode}); see logs/{stage}.log')
            atomic(done,{'finished':time.time(),'code_sha256':code_hashes,'command':command})
            child=None
        result=read(root/'RESULT.json');state(root,'complete',result=result['status'],seconds=time.time()-begin)
    except BaseException as exc:
        state(root,'stopped' if isinstance(exc,(KeyboardInterrupt,TimeoutError)) else 'failed',error=str(exc),seconds=time.time()-begin)
        atomic(root/'ERROR.json',{'error':str(exc),'traceback':traceback.format_exc()})
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        atomic(root/'WORKER.json',{'running':False,'updated':time.time()})
        publish_now()
        lock.close()

if __name__=='__main__':
    p=arguments(__doc__);p.add_argument('--max-hours',type=float,default=9);p.add_argument('--no-publish',action='store_true')
    a=p.parse_args();run(a.root,a.max_hours,not a.no_publish)
