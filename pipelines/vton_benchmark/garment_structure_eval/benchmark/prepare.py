"""Prepare paired manifests and an isolated copy of benchmark inputs."""
import argparse
import importlib.util
import json
import random
import shutil
from pathlib import Path

from .common import HERE, MODELS, CONCEPTS, EXPERT_SHA, digest, atomic, sample_seed


def mask_from_dataset(dataset, iid):
    """Use OOTD's official mask recipe after explicit LIP -> ATR label mapping.

    Input is VITON-HD's supplied LIP parse and BODY_25 keypoints, not RGB
    garment pixels. This common mask protocol is recorded in every report.
    """
    import numpy as np
    from PIL import Image
    path = HERE / 'vendor/ootd/run/utils_ootd.py'
    spec = importlib.util.spec_from_file_location('benchmark_ootd_mask', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lip = Image.open(dataset / 'test/image-parse-v3' / (iid + '.png'))
    w, h = lip.size
    lut = np.array([0, 1, 2, 0, 3, 4, 7, 4, 0, 6, 7, 17, 5, 11, 14, 15, 12, 13, 9, 10], dtype=np.uint8)
    labels = np.asarray(lip)
    if labels.ndim != 2 or labels.max() >= len(lut):
        raise ValueError('Expected indexed LIP parse: ' + iid)
    atr = Image.fromarray(lut[labels])
    people = json.loads((dataset / 'test/openpose_json' / (iid + '_keypoints.json')).read_text())['people']
    if not people:
        raise ValueError('No OpenPose person: ' + iid)
    pose = np.asarray(people[0]['pose_keypoints_2d']).reshape(-1, 3)
    # OOTD uses shoulders/elbows/wrists 2..7, identical in BODY_25 and COCO18.
    xy = pose[:, :2].copy() * [384/w, 512/h]
    xy[pose[:, 2] <= 0] = 0
    mask, _ = mod.get_mask_location('hd', 'upper_body', atr, {'pose_keypoints_2d': xy.reshape(-1).tolist()})
    mask = mask.resize((w, h), Image.Resampling.NEAREST)
    if not np.any(np.asarray(mask)) or np.all(np.asarray(mask)):
        raise ValueError('Degenerate inpainting mask: ' + iid)
    return mask


def prepare(dataset, job, expert):
    dataset, job, expert = Path(dataset).resolve(), Path(job).resolve(), Path(expert).resolve()
    if job.exists():
        raise FileExistsError('Use a fresh job; existing results are immutable.')
    if digest(expert) != EXPERT_SHA:
        raise ValueError('Expert checkpoint differs from the agreed frozen evaluator')
    job.mkdir(parents=True)
    ids = sorted(p.stem for p in (dataset / 'test/image').glob('*.jpg'))
    random.Random(20260924).shuffle(ids)
    manifest = []
    for iid in ids:
        fields = dict(target=f'test/image/{iid}.jpg', garment=f'test/cloth/{iid}.jpg',
                      parse=f'test/image-parse-v3/{iid}.png', densepose=f'test/image-densepose/{iid}.jpg',
                      keypoints=f'test/openpose_json/{iid}_keypoints.json')
        for field, relative in fields.items():
            src, dest = dataset / relative, job / 'data' / relative
            if not src.exists():
                raise FileNotFoundError(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Symlinks locally; the transport archive dereferences only this data tree.
            dest.symlink_to(src)
        manifest.append(dict(id=iid, seed=sample_seed(iid), split='test', **fields,
                             mask=f'test/agnostic-mask/{iid}_mask.png',
                             hashes={k: digest(dataset/v) for k, v in fields.items()}))
    (job / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in manifest))
    shutil.copy2(expert, job / 'experts.pt')
    shutil.copy2(HERE / 'revisions.json', job / 'revisions.json')
    atomic(job / 'config.json', dict(version=1, models=MODELS, concepts=CONCEPTS, width=768, height=1024,
        wall_seconds=43200, report_reserve_seconds=3600, cohorts=[4,64,128,256,512,1024,2032],
        expert_sha256=EXPERT_SHA, manifest_sha256=digest(job/'manifest.jsonl'),
        preprocessing='Shared OOTD mask recipe from dataset LIP parse remapped to ATR and supplied OpenPose; native DensePose; FastFit uses DWPose.',
        output='Pre-RGB-compositing PNG. Preserve model-internal latent inpainting.',
        score='Frozen expert L2 at fixed GT points; independent localization secondary. Not accuracy.',
        inference_only=True))
    atomic(job / 'status.json', dict(state='prepared', launched=False, test_images=len(ids)))
    print(json.dumps(dict(job=str(job), samples=len(ids), expert_sha256=EXPERT_SHA)))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--job',required=True)
    p.add_argument('--expert', default='/mnt/nvme0/garment_structure_eval/runs/full_final/best.pt')
    a=p.parse_args();prepare(a.dataset,a.job,a.expert)
