"""Reproduce the prepared inputs from the recorded manifest and original VITON-HD.

No relabelling or Molmo call: this reuses the committed automatic proposals.
Requires OOTDiffusion's pinned run/utils_ootd.py for the agnostic-mask recipe.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(dataset, manifest_path, ootd_utils, output):
    manifest = json.loads(manifest_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('ootd_mask_recipe', ootd_utils)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lut = np.array([0, 1, 2, 0, 3, 4, 7, 4, 0, 6, 7, 17, 5, 11, 14, 15, 12, 13, 9, 10], dtype=np.uint8)
    height, width = manifest['image_size']
    for row in manifest['records']:
        iid = row['id']
        folder = output / 'data' / iid
        folder.mkdir(parents=True)
        for name, subdir in [('target', 'image'), ('garment', 'cloth'), ('densepose', 'image-densepose')]:
            src = dataset / subdir / (iid + '.jpg')
            if digest(src) != row['sources'][name]['sha256']:
                raise ValueError('Source hash mismatch: ' + str(src))
            Image.open(src).convert('RGB').resize((width, height), Image.Resampling.LANCZOS).save(folder / (name + '.png'))
        parse = Image.open(dataset / 'image-parse-v3' / (iid + '.png'))
        native_w, native_h = parse.size
        labels = np.asarray(parse)
        if labels.ndim != 2 or labels.max() >= len(lut):
            raise ValueError('Expected indexed LIP parsing')
        people = json.loads((dataset / 'openpose_json' / (iid + '_keypoints.json')).read_text())['people']
        pose = np.asarray(people[0]['pose_keypoints_2d']).reshape(-1, 3)
        xy = pose[:, :2].copy() * [384 / native_w, 512 / native_h]
        xy[pose[:, 2] <= 0] = 0
        mask, _ = mod.get_mask_location('hd', 'upper_body', Image.fromarray(lut[labels]),
                                       {'pose_keypoints_2d': xy.reshape(-1).tolist()})
        mask.resize((native_w, native_h), Image.Resampling.NEAREST).resize(
            (width, height), Image.Resampling.NEAREST).save(folder / 'mask.png')
    shutil.copy2(manifest_path, output / 'manifest.json')
    print('PREPARED', len(manifest['records']), 'real paired images in', output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ['dataset', 'manifest', 'ootd-utils', 'output']:
        p.add_argument('--' + arg, required=True, type=Path)
    args = p.parse_args()
    prepare(args.dataset, args.manifest, args.ootd_utils, args.output)
