"""Versioned, frozen-generator localization audit; no training entry point."""
from pathlib import Path
import hashlib
import json
import os
import time

HERE=Path(__file__).resolve().parent
WORKSPACE=Path(os.environ.get('LEFFA_WORKSPACE',str(HERE.parents[1])))
PARTS=('collar','neckline','cuff','sleeve_opening')
TIMESTEPS=(99,299,499,699,899)
SIZE=(384,512)
GRID=(64,48)
SEED=20260929
GPU='GPU-dbb7940e-7461-387e-c0c2-a2e33f9b78ac'
DEFAULT_ROOT=Path('/mnt/nvme0/leffa_native/localization_v2')
DEFAULT_OLD=Path('/mnt/nvme0/leffa_native/20260928_v1')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def state(root,stage,**kwargs):
    atomic(root/'STATE.json',{'stage':stage,'updated':time.time(),'gpu':GPU,'training_enabled':False,**kwargs})
def sha_order(key):return hashlib.sha256(('localization-v2:'+key).encode()).hexdigest()
def manifest(root):return read(root/'manifest.json')['records']
def bind_gpu():
    os.environ['CUDA_VISIBLE_DEVICES']=GPU;os.environ['LEFFA_EXPECTED_GPU']=GPU
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='4'
def cutter_path():
    candidates=[WORKSPACE/'super_garment_net/flatlay_cutter',HERE.parent/'flatlay_cutter']
    for p in candidates:
        if (p/'flatlay_cutter/engine.py').exists():return p
    raise FileNotFoundError('Install the adjacent flatlay_cutter package or use the workspace copy')

def arguments(description):
    import argparse
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    return p
