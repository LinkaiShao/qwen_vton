"""Single owned experiment. Wait for shared GPU capacity; never kill other jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from prepare import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--code',type=Path);p.add_argument('--gpu',help='Authorized GPU UUID; also restricts CUDA visibility')
    p.add_argument('--hours',type=float,default=23)
    for name in ['leffa-code','leffa-weights','dino-weights']:p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.gpu:os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,LEFFA_EXPECTED_GPU=a.gpu)
    root=a.root;start=time.time();deadline=start+a.hours*3600
    lock=(root/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code=a.code or root/'code';run=root/'run';run.mkdir(exist_ok=True)
    home=Path.home()
    common=['--data',str(root),'--cache',str(root/'cache'),'--output',str(run),
            '--leffa-code',str(a.leffa_code or home/'vton_region_benchmark/20260924_eight_models/code/garment_structure_eval/benchmark/vendor/leffa'),
            '--leffa-weights',str(a.leffa_weights or home/'sgn/hf/hub/models--franciszzj--Leffa/snapshots/61d3390f444506f052feedb0b243cd5369c29c89'),
            '--dino-weights',str(a.dino_weights or home/'sgn/hf/hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415')]
    selection=['-i',a.gpu] if a.gpu else []
    inventory=subprocess.run(['nvidia-smi',*selection,'--query-gpu=name,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip().splitlines()
    if len(inventory)!=1:raise RuntimeError('Select one GPU explicitly on multi-GPU machines')
    device=inventory[0]
    active=None

    def stop_owned_child(signum,frame):
        if active is not None and active.poll() is None:
            active.terminate()
            try:active.wait(timeout=30)
            except subprocess.TimeoutExpired:active.kill();active.wait()
        raise SystemExit('Supervisor stopped by signal '+str(signum))

    signal.signal(signal.SIGTERM,stop_owned_child)
    signal.signal(signal.SIGINT,stop_owned_child)

    def state(stage,**extra):
        obj={'stage':stage,'updated':time.time(),'started':start,'deadline':deadline,'gpu':device,**extra}
        atomic(root/'STATE.json',obj);print(json.dumps(obj),flush=True)

    def execute(stage,argv):
        nonlocal active
        state(stage,command=argv)
        with (root/(stage+'.log')).open('a') as log:
            proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT);active=proc
            atomic(root/'WORKER.json',{'pid':proc.pid,'stage':stage,'started':time.time()})
            while proc.poll() is None:
                if time.time()>deadline:
                    proc.terminate();proc.wait(timeout=45)
                    raise TimeoutError('Session deadline; resume from saved checkpoints')
                time.sleep(10)
            if proc.returncode:raise RuntimeError(stage+' failed; see '+str(root/(stage+'.log')))
            active=None

    try:
        execute('cpu_checks',[sys.executable,str(code/'test_core.py')])
        execute('architecture_checks',[sys.executable,str(code/'verify_architecture.py'),
                '--leffa-code',common[common.index('--leffa-code')+1],
                '--leffa-weights',common[common.index('--leffa-weights')+1],
                '--output',str(root/'ARCHITECTURE_VERIFICATION.json')])
        # Install only the independent scorer into a private directory. The
        # existing LeFFA environment and its pinned torch are never upgraded.
        execute('evaluation_dependency',[sys.executable,'-m','pip','install','--no-deps','--target',str(root/'deps'),'lpips==0.1.4'])
        if (root/'dataset.tar').exists() and not ((root/'DATA_READY.json').exists() and (root/'data/train_00000_00/target.png').exists()):
            execute('restore_data',[sys.executable,'-u',str(code/'transfer.py'),'receive','--root',str(root)])
        while True:
            if time.time()>deadline:raise TimeoutError('No selected GPU capacity/data before session deadline')
            data_ready=(root/'DATA_READY.json').exists() and (root/'data/train_00000_00/target.png').exists()
            r=subprocess.run(['nvidia-smi',*selection,'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            free=int(r.stdout.strip().splitlines()[0])
            if data_ready and free>=24576:break
            state('waiting_for_gpu' if data_ready else 'waiting_for_data',free_gpu_mib=free,required_free_mib=24576,data_ready=data_ready)
            time.sleep(30)
        if not (root/'cache/BOOTSTRAP_READY.json').exists():execute('cache_bootstrap',[sys.executable,'-u',str(code/'runner.py'),'cache',*common,'--cache-bootstrap'])
        execute('verify',[sys.executable,'-u',str(code/'runner.py'),'verify',*common])
        if not (root/'cache/READY.json').exists():execute('cache',[sys.executable,'-u',str(code/'runner.py'),'cache',*common])
        if not (run/'TIMESTEP_VERIFICATION.json').exists():execute('sweep',[sys.executable,'-u',str(code/'runner.py'),'sweep',*common])
        execute('train',[sys.executable,'-u',str(code/'runner.py'),'train',*common])
        execute('evaluate',[sys.executable,'-u',str(code/'evaluate.py'),*common])
        execute('report',[sys.executable,str(code/'report.py'),'--root',str(root)])
        state('completed',result=json.loads((run/'RESULT.json').read_text()))
    except BaseException as exc:
        state('failed',error=str(exc));raise


if __name__=='__main__':main()
