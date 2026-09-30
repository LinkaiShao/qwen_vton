"""Deterministic pilot protocol, provenance guards and resumable state."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .schema import digest, usable, validate

FAMILIES = ('buttons', 'graphics', 'patterns', 'construction')


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def check_partitions(rows, replay):
    validate(rows + [r for r in replay if r['id'] not in {x['id'] for x in rows}])
    for split in ('train', 'calibration', 'test'):
        group = [r for r in rows if r['split'] == split]
        if len(group) != 8 or Counter(r['family'] for r in group) != Counter({k: 2 for k in FAMILIES}):
            raise ValueError(f"Need 8 balanced garments in {split}")
    if len(replay) != 512 or any(r['split'] != 'train' or r['dataset_split'] != 'train' for r in replay):
        raise ValueError("Replay must be the original 512 training examples")
    heldout = {r['garment_id'] for r in rows if r['split'] != 'train'}
    if heldout.intersection(r['garment_id'] for r in replay):
        raise ValueError("Held-out garment occurs in replay")
    train = [r for r in rows if r['split'] == 'train']
    if any(r['dataset_split'] != 'train' for r in train):
        raise ValueError("Dataset test targets cannot train")
    if len({r['garment_id'] for r in rows}) != 24:
        raise ValueError("Pilot garments must be distinct")


def require_review(rows, manifest, receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get('manifest_sha256') != digest(manifest) or receipt.get('verified_by_user') is not True:
        raise ValueError("User verification receipt missing or stale")
    for row in rows:
        eligible = [p for p in row['parts'] if usable(p)]
        if not eligible:
            raise ValueError(f"No verified visible region: {row['id']}; replace/review this case")
        for p in eligible:
            if p.get('vae_assessment') not in ('preserved', 'limited'):
                raise ValueError(f"VAE detail assessment is pending: {row['id']}/{p['id']}")
            if not p.get('expected_correction', '').strip():
                raise ValueError("Verified regions need an explicit expected correction")
            if p.get('target_sha256') != digest(row['target']):
                raise ValueError("GT changed after review")
    return receipt


def sample_trace(train_ids, replay_ids, seed=90222, steps=100):
    if len(set(train_ids)) != 8 or len(set(replay_ids)) != 512:
        raise ValueError("Expected eight detail garments and 512 replay garments")
    rng = random.Random(seed); detail = []; trace = []
    for i in range(steps):
        if i % 4 != 3:
            if not detail:
                detail = sorted(train_ids); rng.shuffle(detail)
            iid, source = detail.pop(), 'detail'
        else:
            iid, source = rng.choice(sorted(replay_ids)), 'replay'
        trace.append(dict(step=i+1, id=iid, source=source, seed=rng.randrange(2**31)))
    return trace


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, model, optimizer, step, elapsed, protocol):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    params = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    value = dict(trainables=params, optimizer=optimizer.state_dict(), step=step, sampler_cursor=step,
                 elapsed=elapsed, rng=rng_state(), protocol=protocol)
    tmp = path.with_suffix('.tmp'); torch.save(value, tmp); tmp.replace(path)


def load_checkpoint(path, model, optimizer, protocol):
    # Only load our own local checkpoint; includes Python/NumPy RNG tuples.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt['protocol'] != protocol or ckpt['step'] != ckpt['sampler_cursor']:
        raise ValueError("Resume protocol/trace/review mismatch")
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if params.keys() != ckpt['trainables'].keys():
        raise ValueError("Trainable parameter set changed")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(ckpt['trainables'][n].to(p))
    optimizer.load_state_dict(ckpt['optimizer']); restore_rng(ckpt['rng'])
    return ckpt['step'], ckpt['elapsed']


def pcgrad(loss, groups, params):
    """Same three-task projection policy, plus garment-retention telemetry."""
    def grad(x):
        values = torch.autograd.grad(x, params, retain_graph=True, allow_unused=True)
        return [g.detach().float() if g is not None else torch.zeros_like(p, dtype=torch.float32)
                for g, p in zip(values, params)]
    tasks = [grad(groups[k]) for k in ('g', 's', 'b')]
    rest = grad(groups['rest'])
    projected = [[g.clone() for g in task] for task in tasks]
    def dot(a, b):
        return sum((x*y).sum() for x, y in zip(a, b))
    conflicts = 0
    for i in range(3):
        order = [j for j in range(3) if j != i]; random.shuffle(order)
        for j in order:
            d = dot(projected[i], tasks[j])
            if d < 0:
                conflicts += 1
                coefficient = d / (dot(tasks[j], tasks[j]) + 1e-12)
                projected[i] = [x - coefficient*y for x, y in zip(projected[i], tasks[j])]
    before = dot(tasks[0], tasks[0]).sqrt().item()
    after = dot(projected[0], projected[0]).sqrt().item()
    for k, p in enumerate(params):
        p.grad = (sum(task[k] for task in projected) + rest[k]).to(p.dtype)
    return dict(conflicts=conflicts, garment_norm_pre_pcgrad=before,
                garment_norm_post_pcgrad=after, garment_gradient_retention=after / max(before, 1e-12))


def acceptance(reviews, metrics, expected_ids):
    if len(reviews) != 8 or {r['id'] for r in reviews} != set(expected_ids):
        raise ValueError("Review every validation garment exactly once")
    if any(r.get('outcome') not in ('fixed', 'improved', 'unchanged', 'worse') for r in reviews):
        raise ValueError("Missing blinded semantic outcome")
    if any(type(r.get('major_regression')) is not bool for r in reviews):
        raise ValueError("Missing identity/background/construction regression judgment")
    wins = sum(r['outcome'] in ('fixed', 'improved') for r in reviews)
    clear = any(r['outcome'] == 'fixed' and r['family'] in ('buttons', 'graphics') for r in reviews)
    regressions = sum(r['major_regression'] for r in reviews)
    checks = {k: metrics[k]['B'] <= 1.02*metrics[k]['A'] for k in ('skin', 'background')}
    return dict(passed=wins >= 3 and clear and not regressions and all(checks.values()),
                improved_or_fixed=wins, clear_button_or_graphic_fix=clear,
                major_regressions=regressions, preservation_checks=checks,
                counts=dict(Counter(r['outcome'] for r in reviews)), denominator=8)
