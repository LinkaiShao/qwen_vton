from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONCEPTS = ['collar', 'lapel', 'cuff', 'button hole', 'button', 'pocket', 'zipper', 'sleeve', 'shoulder', 'logo']
EXPERT_SHA = '88373a448b4bf91d47d18b068e8872043489a042321272f659ebd4eaf4e22fc6'
MOLMO_REVISION = '188130f961c8e0888a34e11121a1423c461a01ba'
MODELS = {
    'coral': dict(name='CORAL', repo='cvlab-kaist/CORAL', weights='chimaharicox/coral_vt', steps=28, cfg=30., diffusers='0.35.1', transformers='4.56.2'),
    'deco': dict(name='DeCo-VTON', repo='Levinna/DeCo-VTON', weights='levinna/DeCo-VTON', subfolder='VITON-HD-1024/unet', steps=50, cfg=2.5, diffusers='0.35.1', transformers='4.46.3'),
    'fastfit': dict(name='FastFit', repo='Zheng-Chong/FastFit', weights='zhengchong/FastFit-SR-1024', steps=50, cfg=2.5, diffusers='0.35.1', transformers='4.46.3'),
    'leffa': dict(name='Leffa', repo='franciszzj/Leffa', weights='franciszzj/Leffa', steps=50, cfg=2.5, diffusers='0.31.0', transformers='4.46.3'),
    'catflux': dict(name='CatVTON-FLUX', repo='nftblackmagic/catvton-flux', weights='xiaozaa/catvton-flux-alpha', steps=50, cfg=30., diffusers='0.35.1', transformers='4.46.3'),
    'catvton': dict(name='CatVTON', repo='Zheng-Chong/CatVTON', weights='zhengchong/CatVTON', steps=50, cfg=2.5, diffusers='0.31.0', transformers='4.46.3'),
    'idm': dict(name='IDM-VTON', repo='yisol/IDM-VTON', weights='yisol/IDM-VTON', steps=30, cfg=2., diffusers='0.25.0', transformers='4.36.2'),
    'ootd': dict(name='OOTDiffusion', repo='levihsu/OOTDiffusion', weights='levihsu/OOTDiffusion', steps=20, cfg=2., diffusers='0.24.0', transformers='4.36.2'),
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text())


def rows(job):
    return [json.loads(s) for s in (Path(job) / 'manifest.jsonl').read_text().splitlines() if s.strip()]


def sample_seed(iid):
    return int.from_bytes(hashlib.sha256(('vton-region-v1:' + iid).encode()).digest()[:4], 'big') % (2**31)


def prediction(job, model, iid):
    return Path(job) / 'predictions' / model / (iid + '.png')


def paired_common(records, models):
    """Only compare exactly the same images; incomplete models never vanish."""
    sets = [{r['id'] for r in records if r['model'] == m} for m in models]
    return sorted(set.intersection(*sets)) if sets else []


def next_cohort(current, total, remaining, seconds_per_image, reload_seconds):
    import math
    target = 64 if current <= 4 else min(current * 2, total)
    target = min(target, total)
    # A large next cohort failing to fit must not discard hours of usable time.
    capacity = math.floor(((remaining-3600)/1.25-reload_seconds)/max(1,seconds_per_image))
    target = min(target, current+max(0,capacity))
    return target if target > current else None


def estimate_image_seconds(elapsed, gpu_wait, model_loads, delta, inference, grounding, limit):
    # GPU contention consumes the wall budget, but is not work to extrapolate
    # for each future image. Cached generation still supplies a lower bound.
    active = max(1, elapsed-gpu_wait-model_loads)/max(1,delta)
    return max(active, inference+grounding/max(1,limit))
