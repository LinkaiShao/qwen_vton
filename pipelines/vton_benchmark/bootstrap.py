"""Fetch pinned public model sources and checksum-verified HR-VITON weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
REPOS={
    'hrviton':('sangyun884/HR-VITON','2715bdd687b3a07b8b8bcb3e44aa5533b28c1f15'),
    'ootd':('levihsu/OOTDiffusion','13ef0faba266cdde9febc8ad39be2395bbb89d9c'),
}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True,type=Path)
    parser.add_argument('--sources-only',action='store_true')
    args=parser.parse_args()
    vendor=ROOT/'garment_structure_eval/benchmark/vendor'
    vendor.mkdir(parents=True,exist_ok=True)
    for name,(repo,revision) in REPOS.items():
        dest=vendor/name
        if not dest.exists():
            subprocess.run(['git','clone','--no-checkout',f'https://github.com/{repo}.git',str(dest)],check=True)
            subprocess.run(['git','-C',str(dest),'checkout','--detach',revision],check=True)
        actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(dest),'status','--porcelain','--untracked-files=no'],text=True)
        if actual!=revision or dirty:
            raise RuntimeError(f'{dest} differs from the pinned source; use a clean checkout.')
        print(name,revision,flush=True)
    if args.sources_only:return
    import gdown
    from huggingface_hub import snapshot_download
    provenance=json.loads((ROOT/'run_reference/provenance.json').read_text())
    checkpoints=args.job/'checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
    for name,spec in provenance['weights'].items():
        path=checkpoints/name
        if not path.exists():
            temporary=path.with_suffix('.download')
            gdown.download(id=spec['google_drive_id'],output=str(temporary),quiet=False)
            if sha(temporary)!=spec['sha256']:raise RuntimeError(f'Checkpoint hash mismatch: {name}')
            temporary.replace(path)
        if sha(path)!=spec['sha256']:raise RuntimeError(f'Checkpoint hash mismatch: {name}')
        print('Verified',name,flush=True)
    snapshot_download('facebook/dinov2-base',revision=provenance['dino_revision'],
        allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])


if __name__=='__main__':main()
