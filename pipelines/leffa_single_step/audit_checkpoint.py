"""Audit optimizer steps, paired training schedules and real projection updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def audit(run, initial_weights):
    init = torch.load(initial_weights, map_location='cpu', weights_only=True)
    arms = ['denoise_spatial_control', 'denoise_spatial_dino']
    logs = [[json.loads(line) for line in (run / (arm + '_train.jsonl')).read_text().splitlines()] for arm in arms]
    assert [(r['id'], r['timestep']) for r in logs[0]] == [(r['id'], r['timestep']) for r in logs[1]], 'Unmatched arm schedules'
    summary = {'paired_image_and_timestep_schedules_identical': True, 'arms': {}}
    for arm, rows in zip(arms, logs):
        checkpoint = torch.load(run / (arm + '_latest.pt'), map_location='cpu', weights_only=True)
        assert [r['step'] for r in rows] == list(range(1, len(rows) + 1))
        assert checkpoint['step'] == len(rows)
        steps = {int(state['step']) for state in checkpoint['optimizer']['state'].values()}
        assert steps == {len(rows)}, steps
        names = ['unet.' + layer + '.' + projection + '.weight' for layer in checkpoint['layers'] for projection in ['to_q', 'to_k']]
        updates = []
        for name, trained in zip(names, checkpoint['projection_parameters']):
            # load_models first casts the frozen model to BF16, then promotes
            # trainable projections to FP32. Compare with that exact starting value.
            original = init[name].to(torch.bfloat16).float()
            difference = trained.float() - original
            norm = float(difference.norm())
            assert torch.isfinite(difference).all() and norm > 0, name
            updates.append({'parameter': name, 'elements': trained.numel(), 'update_l2': norm,
                            'relative_update_l2': norm / float(original.norm()),
                            'update_max_abs': float(difference.abs().max())})
        times = [b['elapsed_seconds'] - a['elapsed_seconds'] for a,b in zip(rows, rows[1:]) if b['step'] % 100 != 1]
        summary['arms'][arm] = {'updates': len(rows), 'optimizer_steps': sorted(steps),
                               'projection_updates': updates,
                               'median_seconds_per_update_excluding_evaluation': float(np.median(times)),
                               'peak_cuda_gib': max(r['peak_cuda_gib'] for r in rows),
                               'first_50_mean_training_mass': 1 - float(np.mean([r['loss_spatial'] for r in rows[:50]])),
                               'last_50_mean_training_mass': 1 - float(np.mean([r['loss_spatial'] for r in rows[-50:]]))}
    (run / 'CHECKPOINT_AUDIT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--initial-weights', type=Path, required=True)
    a = p.parse_args();audit(a.run, a.initial_weights)
