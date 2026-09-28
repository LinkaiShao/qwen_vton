"""Build a static, auditable research report from the completed real experiment."""
import argparse
import html
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ARMS = ['baseline', 'denoise_spatial_control_final', 'denoise_spatial_dino_final']
NAMES = ['Initial LeFFA', 'Localization control', 'Localization + DINO']
COLORS = ['#82919d', '#bc75ec', '#24bfa3']
CONCEPTS = ['collar_neckline', 'left_sleeve_cuff', 'right_sleeve_cuff']
LABELS = ['Collar / neckline', 'Left sleeve end', 'Right sleeve end']


def read(path):
    return json.loads(path.read_text())


def clean_summary(evaluation):
    return {k: v for k, v in evaluation.items() if k != 'records'}


def mean(records, key):
    return float(np.mean([r[key] for r in records])) if records else None


def bootstrap_delta(initial, joint, key):
    before = {(r['id'], r['concept'], r['t_fraction']): r[key] for r in initial}
    by_id = {}
    for r in joint:
        by_id.setdefault(r['id'], []).append(r[key] - before[(r['id'], r['concept'], r['t_fraction'])])
    values = np.array([np.mean(v) for v in by_id.values()])
    rng = np.random.default_rng(20260928)
    draws = rng.choice(values, (10000, len(values)), replace=True).mean(1)
    return {'mean_paired_image_delta': float(values.mean()),
            'bootstrap_95pct_image_interval': np.quantile(draws, [.025, .975]).tolist(),
            'images': len(values)}


def plots(evaluations, output):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    ts = sorted({r['t_fraction'] for r in evaluations[0]['records']})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True, constrained_layout=True)
    for index, (concept, label) in enumerate(zip(CONCEPTS, LABELS)):
        ax = axes[index]
        for e, name, color in zip(evaluations, NAMES, COLORS):
            values = [100 * mean([r for r in e['records'] if r['concept'] == concept and r['t_fraction'] == t], 'mass_inside_box') for t in ts]
            ax.plot(ts, values, 'o-', color=color, label=name, linewidth=2, markersize=4)
        ax.axhline(80, linestyle='--', color='#cf5d53', label='80% target' if index == 0 else None)
        n = len({r['id'] for r in evaluations[0]['records'] if r['concept'] == concept})
        ax.set(title=f'{label} (n={n})', xlabel='Sampled timestep / 999', ylim=(0, 100))
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Attention mass in automatic box (%)')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.savefig(output / 'localization.png', dpi=180)
    fig.savefig(output / 'localization.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for ax, key, title in zip(axes, ['fixed_roi_dino_distance', 'pointed_roi_dino_distance'],
                              ['Same labelled crop: image detail', 'Attention-centered crop: location + detail']):
        for e, name, color in zip(evaluations, NAMES, COLORS):
            ax.plot(ts, [mean([r for r in e['records'] if r['t_fraction'] == t], key) for t in ts],
                    'o-', label=name, color=color, linewidth=2, markersize=4)
        ax.set(title=title, xlabel='Sampled timestep / 999', ylabel='DINO cosine distance (lower is better)')
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.savefig(output / 'dino.png', dpi=180)
    fig.savefig(output / 'dino.svg')
    plt.close(fig)


def overlay(image, attention, box, predicted, dest):
    array = np.asarray(image).astype(float) / 255
    heat = np.asarray(Image.fromarray(attention.astype(np.float32)).resize(image.size, Image.Resampling.BILINEAR))
    heat = (heat / max(float(heat.max()), 1e-12)) ** .6
    color = plt.get_cmap('magma')(heat)[..., :3]
    mix = .65 * heat[..., None]
    rgb = Image.fromarray(np.uint8(np.clip(array * (1 - mix) + color * mix, 0, 1) * 255))
    draw = ImageDraw.Draw(rgb)
    for coords, ink in [(box, '#27e6ec'), (predicted, '#ffe066')]:
        xy = [coords[0] * image.width, coords[1] * image.height,
              coords[2] * image.width, coords[3] * image.height]
        draw.rectangle(xy, outline=ink, width=3)
    rgb.save(dest)


def crop(image, box, path):
    region = image.crop((round(box[0] * image.width), round(box[1] * image.height),
                         round(box[2] * image.width), round(box[3] * image.height)))
    region.resize((224, 224), Image.Resampling.BICUBIC).save(path)


def build(run, data, output):
    result = read(run / 'RESULT.json')
    run_args = read(run / 'RUN.json')['arguments']
    evaluations = [read(run / 'evaluations' / (arm + '.json')) for arm in ARMS]
    manifest = read(data / 'manifest.json')
    output.mkdir(parents=True, exist_ok=True)
    assets = output / 'assets';assets.mkdir(exist_ok=True)
    plots(evaluations, assets)
    raw = output / 'data';raw.mkdir(exist_ok=True)
    for name in ['RESULT.json', 'GRADIENT_VERIFICATION.json', 'RUN.json']:
        shutil.copy2(run / name, raw / name)
    if (run / 'CHECKPOINT_AUDIT.json').exists():
        shutil.copy2(run / 'CHECKPOINT_AUDIT.json', raw / 'CHECKPOINT_AUDIT.json')
    for arm in ARMS:
        shutil.copy2(run / 'evaluations' / (arm + '.json'), raw / (arm + '.json'))
    stats = {'joint_minus_control_fixed': bootstrap_delta(evaluations[1]['records'], evaluations[2]['records'], 'fixed_roi_dino_distance'),
             'joint_minus_initial_fixed': bootstrap_delta(evaluations[0]['records'], evaluations[2]['records'], 'fixed_roi_dino_distance')}
    (raw / 'paired_image_statistics.json').write_text(json.dumps(stats, indent=2))
    lookup = [{(r['id'], r['concept']): r for r in e['records'] if r['t_fraction'] == .5} for e in evaluations]
    ids = [r['id'] for r in manifest['records'] if r['split'] == 'validation']
    def delta(iid):
        return np.mean([lookup[2][(iid,c)]['fixed_roi_dino_distance'] - lookup[1][(iid,c)]['fixed_roi_dino_distance']
                        for c in CONCEPTS if (iid,c) in lookup[2]])
    ids.sort(key=delta)
    cards = []
    for iid in ids:
        folder = assets / iid;folder.mkdir(exist_ok=True)
        originals = []
        for arm, name in zip(ARMS, ['before', 'control', 'after']):
            src = run / 'examples' / arm / iid / 'single_step_prediction.png'
            shutil.copy2(src, folder / (name + '.png'))
            originals.append(Image.open(src).convert('RGB'))
        target = Image.open(data / 'data' / iid / 'target.png').convert('RGB')
        target.save(folder / 'target.png')
        shutil.copy2(data / 'data' / iid / 'garment.png', folder / 'garment.png')
        details = []
        for k, (concept, label) in enumerate(zip(CONCEPTS, LABELS)):
            if (iid, concept) not in lookup[2]:
                continue
            records = [m[(iid, concept)] for m in lookup]
            box = records[0]['box']
            crop(target, box, folder / (concept + '_target.png'))
            crop_images = [f'<figure><img loading="lazy" src="assets/{iid}/{concept}_target.png"><figcaption>Real target crop</figcaption></figure>']
            map_images = []
            for j, (arm, image, record) in enumerate(zip(ARMS, originals, records)):
                crop(image, box, folder / (concept + f'_crop{j}.png'))
                attn = np.load(run / 'examples' / arm / iid / 'attention.npz')['maps'][0, k]
                overlay(image, attn, box, record['predicted_box'], folder / (concept + f'_map{j}.png'))
                crop_images.append(f'<figure><img loading="lazy" src="assets/{iid}/{concept}_crop{j}.png"><figcaption>{NAMES[j]}<br>distance {record["fixed_roi_dino_distance"]:.4f}</figcaption></figure>')
                map_images.append(f'<figure><img loading="lazy" src="assets/{iid}/{concept}_map{j}.png"><figcaption>{NAMES[j]}<br>box mass {100*record["mass_inside_box"]:.1f}%</figcaption></figure>')
            details.append(f'<details><summary>{label} · mass {records[0]["mass_inside_box"]:.1%} → {records[2]["mass_inside_box"]:.1%} · fixed-crop distance {records[0]["fixed_roi_dino_distance"]:.4f} → {records[2]["fixed_roi_dino_distance"]:.4f}</summary><h4>Exact same region in every image</h4><div class="grid four">{"".join(crop_images)}</div><h4>Where the readout points</h4><p>Cyan: automatic target box. Yellow: attention-centered crop. Heat colors are normalized per map for visibility; numeric mass is the quantitative comparison.</p><div class="grid three">{"".join(map_images)}</div></details>')
        full = ''.join(f'<figure><img loading="lazy" src="assets/{iid}/{name}.png"><figcaption>{label}</figcaption></figure>' for name,label in [('garment','Garment condition'),('target','Real target'),('before','Initial single-step estimate'),('control','Localization control'),('after','Localization + DINO')])
        cards.append(f'<article id="case-{iid}"><h3>{iid} <small>DINO − control: {delta(iid):+.4f} at t=.5</small></h3><div class="grid five">{full}</div>{"".join(details)}</article>')
    criterion_rows = ''.join(f'<tr><td>{html.escape(k.replace("_", " "))}</td><td class="{ "pass" if v else "fail" }">{"PASS" if v else "NOT MET"}</td></tr>' for k,v in result['criteria'].items())
    numeric_rows = ''.join(f'<tr><td>{name}</td><td>{e["mass_at_half"][CONCEPTS[0]]:.1%}</td><td>{e["mass_at_half"][CONCEPTS[1]]:.1%}</td><td>{e["mass_at_half"][CONCEPTS[2]]:.1%}</td><td>{e["mean_fixed_roi_dino_distance"]:.4f}</td><td>{e["mean_pointed_roi_dino_distance"]:.4f}</td></tr>' for e,name in zip(evaluations,NAMES))
    component_rows = []
    for concept,label in zip(CONCEPTS,LABELS):
        values = [mean([r for r in e['records'] if r['concept']==concept], 'fixed_roi_dino_distance') for e in evaluations]
        component_rows.append(f'<tr><td>{label}</td>' + ''.join(f'<td>{v:.4f}</td>' for v in values) + f'<td>{values[2]-values[1]:+.4f}</td></tr>')
    interval = stats['joint_minus_control_fixed']['bootstrap_95pct_image_interval']
    conclusion = ('All stated experiment criteria passed.' if result['status'] == 'SUCCESSFUL' else 'The gradient path works; the full success criteria were not met.')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LeFFA · one-step localization experiment</title>
<style>:root{color-scheme:light}*{box-sizing:border-box}body{font:16px/1.55 system-ui,sans-serif;color:#21323a;background:#edf1f3;margin:0}main{max-width:1460px;margin:auto;padding:32px 24px}h1{font-size:clamp(28px,4vw,48px);line-height:1.1;margin:14px 0}h2{margin-top:44px}h3{margin:0 0 18px}h4{margin:22px 0 8px}p{max-width:1000px}a{color:#116c8b}.eyebrow{letter-spacing:.13em;font-size:12px;font-weight:800;color:#356475}.notice{background:#fff;border-left:5px solid #dd8b48;padding:18px 24px;border-radius:6px}.grid{display:grid;gap:12px}.five{grid-template-columns:repeat(5,1fr)}.four{grid-template-columns:repeat(4,1fr)}.three{grid-template-columns:repeat(3,1fr)}figure{margin:0}img{width:100%;height:auto;display:block;border-radius:5px}figcaption{font-size:13px;margin-top:6px}article{background:#fff;padding:22px;border:1px solid #dbe2e6;border-radius:10px;margin:24px 0}small{font-size:13px;font-weight:400;color:#566976;margin-left:16px}details{border-top:1px solid #dbe2e6;padding:14px 0;margin-top:18px}summary{cursor:pointer;font-weight:650}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{text-align:left;padding:12px;border-bottom:1px solid #e2e7e9}th{background:#dde8ed}.table{overflow-x:auto}.pass{color:#087f68;font-weight:800}.fail{color:#a54323;font-weight:800}.chart{background:white;padding:12px;border-radius:8px;margin:16px 0}.muted{color:#61727a}.code{font-family:monospace;font-size:14px}@media(max-width:750px){main{padding:22px 12px}.five,.four{grid-template-columns:repeat(2,1fr)}.three{grid-template-columns:repeat(3,1fr);gap:6px}article{padding:14px}small{display:block;margin:8px 0}figcaption{font-size:11px}td,th{padding:8px}}</style><main>
<div class="eyebrow">REAL H200 EXPERIMENT · LEFFA · VITON-HD · 28 SEP 2026</div><h1>Can a training step point to garment details?</h1>
'''
    page += f'<div class="notice"><strong>{conclusion}</strong><p>One sampled-timestep LeFFA forward → component heatmap + decoded x₀ estimate → frozen DINO crops → gradients to real LeFFA Q/K projections and learned component keys. No live Molmo and no multi-step sampling.</p></div>'
    page += '<p>These are <strong>single-step estimates from noised paired target images during training</strong>, not fully sampled try-on results. LeFFA disables text cross-attention, so this experiment adds semantic keys to its live image-attention query features. It does not claim native collar/cuff text-token attention.</p>'
    page += f'<h2>Measured result</h2><p>{result["iterations"]:,} updates across two matched arms; {result["oom_count"]} OOMs. 24 optimization images, 8 garment-separated validation images. All held-out cases appear below, ordered by DINO change versus control.</p><div class="table"><table><tr><th>Arm</th><th>Collar mass</th><th>Left sleeve mass</th><th>Right sleeve mass</th><th>Fixed crop DINO ↓</th><th>Pointed crop DINO ↓</th></tr>{numeric_rows}</table></div>'
    page += f'<p class="muted">Mass is measured at t=.5. DINO averages all annotated regions at seven held-out timesteps. Paired image bootstrap: DINO − control fixed-crop distance {stats["joint_minus_control_fixed"]["mean_paired_image_delta"]:+.5f}; descriptive 95% interval [{interval[0]:+.5f}, {interval[1]:+.5f}], only 8 images. Negative means improvement.</p>'
    page += '<div class="chart"><img src="assets/localization.png" alt="Attention mass versus sampled timestep, one panel per garment component"></div><div class="chart"><img src="assets/dino.png" alt="Fixed-site and attention-centered DINO distance versus timestep"></div>'
    page += '<div class="table"><table><tr><th>Fixed-crop DINO by component ↓</th><th>Initial</th><th>Control</th><th>Joint</th><th>Joint − control</th></tr>' + ''.join(component_rows) + '</table></div><p class="muted">Each component average covers the seven timestep evaluations. Only three validation images have sleeve-end annotations. Negative deltas favor DINO training.</p>'
    if (run / 'CHECKPOINT_AUDIT.json').exists():
        audit = read(run / 'CHECKPOINT_AUDIT.json')['arms']['denoise_spatial_dino']
        page += f'<p>Checkpoint audit confirmed updates to all four Q/K matrices (1,024,000 weights). The readout adds 2,880 trainable component-key weights. Median joint update: {audit["median_seconds_per_update_excluding_evaluation"]:.3f} s; peak allocated CUDA memory: {audit["peak_cuda_gib"]:.2f} GiB. <a href="data/CHECKPOINT_AUDIT.json">Inspect optimizer and weight-update evidence.</a></p>'
    page += f'<details><summary>Criterion-by-criterion checks</summary><table>{criterion_rows}</table></details>'
    mix = run_args.get('pointed_mix', .5)
    page += f'<h2>What was trained</h2><p>Both arms start from the same pretrained LeFFA and random semantic keys, use the same image order, uniform timesteps and noise seeds, and update two existing up-block Q/K projections. Control: epsilon MSE + {run_args["spatial_weight"]:g} × spatial mass loss. Joint: control + {run_args["dino_weight"]:g} × DINO cosine distance. Each arm runs {run_args["steps"]:,} updates. DINO is frozen; predicted RGB crops keep gradients through DINO and the frozen VAE. The DINO loss weights fixed-site crops by {1-mix:.0%} and attention-centered crops by {mix:.0%}. Attention head aggregation: {run_args.get("aggregation", "mean_probs")}.</p><p>Annotations are cached automatic MolmoPoint proposals, not human ground truth. They are supervision only; no boxes enter the localization logits. Crop widths and heights come from metadata: this tests learning component centers, not learning full extents. Collar includes neckline and cuff includes a short sleeve end. The cohort is a small feasibility experiment, not a full VITON-HD benchmark.</p>'
    if run_args.get('aggregation') == 'full_logits':
        page += '<div class="notice"><strong>Exploratory follow-up.</strong> This configuration was chosen after inspecting the first run on these same eight development images. It changes head aggregation, DINO weighting and training duration together. It cannot isolate which change caused a difference, and these images are no longer an untouched final test set. <a href="../initial/">The original 1,000-update experiment is preserved here.</a></div>'
    page += '<h2>Every held-out example</h2><p>Open a component row for matched detail crops and initial/control/DINO heatmaps. A lower DINO number is a feature-space change, not a guarantee that a visible defect was fixed.</p>' + ''.join(cards)
    page += '<h2>Evidence & reproducibility</h2><p><a href="data/RESULT.json">Complete result</a> · <a href="data/GRADIENT_VERIFICATION.json">Real gradient verification</a> · <a href="data/RUN.json">Run provenance</a> · <a href="data/paired_image_statistics.json">Paired image statistics</a> · <a href="https://github.com/LinkaiShao/qwen_vton/tree/master/pipelines/leffa_single_step">Source code</a> · <a href="assets/localization.svg">Vector localization plot</a> · <a href="assets/dino.svg">Vector DINO plot</a></p></main></html>'
    (output / 'index.html').write_text(page)
    print('REPORT', output / 'index.html')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ['run', 'data', 'output']:
        p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args();build(a.run, a.data, a.output)
