"""Publish existing VTON results as GitHub-renderable figures; never rescore."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .publish_profiles import MODELS, REGIONS, LABELS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph(summary, site):
    cells = {(c['model'], c['concept']): c for c in summary['cells']}
    counts = [cells['coral', c]['n'] for c in REGIONS]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'text.color': '#172238',
                         'axes.labelcolor': '#172238'})
    fig, ax = plt.subplots(figsize=(16, 8.4))
    fig.subplots_adjust(left=.075, right=.985, top=.76, bottom=.20)
    for i, n in enumerate(counts):
        if n < 10:
            ax.axvspan(i-.46, i+.46, color='#fff3d8', zorder=0)
    width = .095
    offsets = (np.arange(len(MODELS))-3.5)*width
    for j, (key, (name, color, marker)) in enumerate(MODELS.items()):
        points = [cells[key, c] for c in REGIONS]
        y = np.array([p['mean'] for p in points])
        errors = np.array([[p['mean']-p['low'] for p in points],
                           [p['high']-p['mean'] for p in points]])
        assert np.isfinite(y).all() and (errors >= 0).all()
        assert all(p['n'] == n for p, n in zip(points, counts))
        ax.bar(np.arange(len(REGIONS))+offsets[j], y, width=width*.92,
               yerr=errors, color=color, edgecolor='white', linewidth=.3,
               error_kw=dict(elinewidth=.7, capsize=1.7, ecolor='#334155'),
               label='FastFit-SR' if key == 'fastfit' else name)
    ax.set_xlim(-.53, len(REGIONS)-.47)
    ax.set_ylim(0, .75)
    ax.set_yticks(np.arange(0, .751, .1))
    ax.set_xticks(np.arange(len(REGIONS)), [f'{label}\nn = {n}' for label, n in zip(LABELS, counts)])
    ax.tick_params(axis='x', length=0, pad=12)
    ax.set_ylabel('Mean regional feature distance to GT  ↓', labelpad=12)
    ax.grid(axis='y', color='#e1e6ee', linewidth=.7)
    ax.set_axisbelow(True)
    for x in np.arange(.5, len(REGIONS)-1, 1):
        ax.axvline(x, color='#edf0f5', linewidth=.7)
    fig.text(.075, .94, 'Bar chart · VTON detail distance by location', fontsize=23, weight='bold')
    fig.text(.075, .893, '128 matched VITON-HD images · 8 models · bar heights = means · whiskers = 95% bootstrap intervals', fontsize=13)
    ax.legend(ncol=4, loc='lower left', bbox_to_anchor=(-.012, 1.025),
              frameon=False, fontsize=12, columnspacing=3.4, handletextpad=.6)
    fig.text(.075, .082, 'Lower is closer to GT in the learned feature space. Compare models within each location; this is not detail accuracy.', fontsize=11)
    fig.text(.075, .047, 'Shading: fewer than 10 examples. No eligible GT detections for lapels, buttonholes or zippers. Values are unchanged.', fontsize=11, color='#526075')
    for ext in ['png', 'svg']:
        fig.savefig(site / f'detail_bars.{ext}', dpi=180, facecolor='white')
    plt.close(fig)
    # Same means, same model and location order; only the presentation changes.
    fig, ax = plt.subplots(figsize=(14, 8.4))
    fig.subplots_adjust(left=.16, right=.88, top=.79, bottom=.20)
    matrix = np.array([[cells[m,c]['mean'] for c in REGIONS] for m in MODELS])
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=.30, vmax=.55, aspect='auto')
    ax.set_xticks(np.arange(len(REGIONS)), [f'{label}\nn = {n}' for label,n in zip(LABELS,counts)])
    ax.set_yticks(np.arange(len(MODELS)), ['FastFit-SR' if k == 'fastfit' else v[0] for k,v in MODELS.items()])
    ax.tick_params(length=0, pad=10)
    for i in range(len(MODELS)):
        for j in range(len(REGIONS)):
            v = matrix[i,j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=15,
                    weight='bold', color='white' if v > .465 else '#172238')
    for i in np.arange(.5, len(MODELS), 1):ax.axhline(i,color='white',linewidth=2)
    for j in np.arange(.5, len(REGIONS), 1):ax.axvline(j,color='white',linewidth=2)
    for spine in ax.spines.values():spine.set_visible(False)
    cax=fig.add_axes([.905,.20,.017,.59])
    cb=fig.colorbar(im,cax=cax);cb.set_label('Mean expert distance · lower is better',labelpad=10)
    fig.text(.035,.94,'Heatmap · VTON detail distance by location',fontsize=23,weight='bold')
    fig.text(.035,.892,'Same 128 matched images · same eight models · same scores · lighter cells mean lower distance',fontsize=12)
    fig.text(.035,.085,'Compare models within a column. Buttons: n=8; pockets: n=3. Confidence intervals are shown in the bar chart.',fontsize=11)
    fig.text(.035,.047,'No eligible GT detections for lapels, buttonholes or zippers. Feature distance is not tiny-detail accuracy.',fontsize=11,color='#526075')
    for ext in ['png','svg']:fig.savefig(site/f'detail_heatmap.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    assert (site/'region_profiles.png').exists(), 'Preserve and reuse the original line chart.'
    return cells, counts


FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def put_image(canvas, image, rect):
    x, y, w, h = rect
    im = ImageOps.contain(image, (w, h), Image.Resampling.LANCZOS)
    canvas.paste(im, (x+(w-im.width)//2, y+(h-im.height)//2))


def window(point, size):
    x, y = point
    w, h = size
    gx, gy = min(15, int(x*16)), min(15, int(y*16))
    return tuple(map(int, (max(0, gx-2)*w/16, max(0, gy-2)*h/16,
                           min(16, gx+3)*w/16, min(16, gy+3)*h/16)))


def detail_sheet(job, by, manifest, concept, selected, dest):
    spread, iid, points = selected
    row = manifest[iid]
    gt_path, cloth_path = job/'data'/row['target'], job/'data'/row['garment']
    gt, cloth = Image.open(gt_path).convert('RGB'), Image.open(cloth_path).convert('RGB')
    assert digest(gt_path) == by['coral', iid]['target_sha256']
    box = window(points[0], gt.size)
    w, h = 1144, 1588
    canvas = Image.new('RGB', (w, h), '#f4f6fa')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), f'{concept.title()} — GT and eight VTON models', font=font(29, True), fill='#172238')
    draw.text((24, 61), f'Case {iid} · selected by largest model-score spread · same crop coordinates in every output', font=font(16), fill='#465569')
    draw.text((28, 101), 'Garment reference', font=font(16, True), fill='#172238')
    draw.text((239, 101), 'GT with crop marked', font=font(16, True), fill='#172238')
    draw.text((475, 101), 'Scorer input size', font=font(16, True), fill='#172238')
    marked = gt.copy()
    d = ImageDraw.Draw(marked)
    d.rectangle(box, outline='#00a7cc', width=6)
    for x, y in points:
        px, py = x*gt.width, y*gt.height
        d.ellipse((px-6, py-6, px+6, py+6), fill='#00a7cc', outline='white', width=2)
    put_image(canvas, cloth, (28, 128, 180, 240))
    put_image(canvas, marked, (239, 128, 180, 240))
    # Illustration of the scorer's full-image resize, not a replacement scoring input.
    small = gt.resize((224, 224), Image.Resampling.BICUBIC)
    ds = ImageDraw.Draw(small)
    scaled_box = tuple(int(v/s*224) for v, s in zip(box, [gt.width, gt.height]*2))
    ds.rectangle(scaled_box, outline='#00a7cc', width=2)
    canvas.paste(small, (475, 130))
    note = (f'The crop below is point 1 of {len(points)} scored points. '
            'The numeric score summarizes the component, not just this crop. '
            'Crops use original-resolution pixels; the evaluator used the full resized image. '
            'Lower feature distance is not a verified visual winner.')
    yy = 130
    for line in textwrap.wrap(note, width=37):
        draw.text((740, yy), line, font=font(17), fill='#465569')
        yy += 24
    items = [('GT', gt, None, '#172238')]
    receipts = []
    for key, (name, color, _) in MODELS.items():
        rec = by[key, iid]
        path = job/'collected/predictions'/key/(iid+'.png')
        assert digest(path) == rec['prediction_sha256']
        image = Image.open(path).convert('RGB')
        assert image.size == gt.size
        part = next(p for p in rec['parts'] if p['concept'] == concept)
        assert part['gt_points'][:4] == points
        items.append(('FastFit-SR' if key == 'fastfit' else name, image, part['fixed_distance'], color))
        receipts.append(dict(model=key, distance=part['fixed_distance'], image_sha256=rec['prediction_sha256']))
    for i, (name, im, score, color) in enumerate(items):
        xx, yy = 24+(i%3)*372, 394+(i//3)*386
        draw.rounded_rectangle((xx, yy, xx+352, yy+366), radius=7, fill='white', outline='#d4dce7', width=1)
        draw.rectangle((xx, yy, xx+352, yy+5), fill=color)
        draw.text((xx+12, yy+12), name, font=font(20, True), fill='#172238')
        score_text = 'Reference' if score is None else f'Distance {score:.3f}'
        draw.text((xx+12, yy+42), score_text, font=font(16), fill='#465569')
        put_image(canvas, im.crop(box), (xx+12, yy+73, 328, 280))
    footer = ('Molmo labeled this as collar, but the visible feature is a neckline. Original scores are retained.'
              if concept == 'collar' else
              'Cyan marks automatic GT locations. Location labels are not manually verified annotations.')
    draw.text((24, 1560), footer, font=font(15), fill='#465569')
    canvas.save(dest, optimize=True)
    return dict(concept=concept, case=iid, selection='largest_model_score_spread', spread=spread,
                gt_points=points, crop_point_index=0, crop_box_pixels=box,
                target_sha256=digest(gt_path), models=receipts, image=dest.name)


def export(job, site):
    report = job/'collected/report'
    summary = json.loads((report/'summary.json').read_text())
    assert summary['common_n'] == 128 and summary['models'] == list(MODELS)
    source_hashes = {name: digest(report/name) for name in ['summary.json', 'scores.json', 'scores.csv']}
    records = json.loads((report/'scores.json').read_text())
    by = {(r['model'], r['id']): r for r in records}
    manifest = {r['id']: r for r in map(json.loads, (job/'collected/manifest.jsonl').read_text().splitlines())}
    site.mkdir(parents=True, exist_ok=True)
    images = site/'comparison-images'
    images.mkdir(exist_ok=True)
    cells, counts = graph(summary, site)
    selections = []
    for concept in REGIONS:
        candidates = []
        for iid in summary['common_ids']:
            parts = [next(p for p in by[m, iid]['parts'] if p['concept'] == concept) for m in MODELS]
            if not parts[0]['gt_present']:
                continue
            values = [p['fixed_distance'] for p in parts]
            candidates.append((max(values)-min(values), iid, parts[0]['gt_points'][:4]))
        selected = max(candidates)
        record = detail_sheet(job, by, manifest, concept, selected, images/(concept.replace(' ', '-')+'.png'))
        selections.append(record)
    receipt = dict(source_hashes=source_hashes, scores_changed=False, models_retrained=False,
                   graphs=['Heatmap of unchanged means', 'Grouped bars with 95% bootstrap intervals', 'Original unchanged line chart'],
                   source_scoring='Full-image 224x224 DINO; 16x16 patch grid; 5x5 token windows; up to four points',
                   selections=selections)
    (site/'comparison-images/manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    lines = ['# VTON detail comparison', '',
             'GT versus eight released VTON models on 128 matched VITON-HD test images. The graph and image sheets render directly on GitHub. Click an image to enlarge it.', '',
             '[Architecture report — warping, attention and garment networks](architecture-report.md) · [Training and conditioning table](model-training.md) · [Raw regional scores](scores.csv) · [Image provenance](comparison-images/manifest.json)', '',
             '## Three graph options', '',
             'Every view uses the same means, model order and seven measured locations. **Lower feature distance is better.** Counts are the eligible images for that component, shared by every model. No scores have been changed.', '',
             '### Heatmap', '',
             'Read down a column to compare the eight models at one location. Each cell prints the exact mean.', '',
             '![Heatmap of the same eight VTON models and seven garment locations](detail_heatmap.png)', '',
             '### Bar chart', '',
             'Compare eight model bars within each location. Whiskers show 95% image-bootstrap intervals; the vertical axis starts at zero.', '',
             '![Grouped bars showing eight VTON models at seven locations with confidence intervals](detail_bars.png)', '',
             '### Original line chart', '',
             'The original location-versus-score chart is preserved unchanged. Lines connect category means; confidence intervals are not drawn on this view.', '',
             '![Original eight-model line chart of regional feature distances](region_profiles.png)', '',
             '## Exact means', '',
             '| Model | '+' | '.join(f'{label} (n={n})' for label, n in zip(LABELS, counts))+' |',
             '|---|'+'---:|'*len(REGIONS)]
    for key, (name, _, _) in MODELS.items():
        lines.append('| '+('FastFit-SR' if key == 'fastfit' else name)+' | '+' | '.join(f'{cells[key,c]["mean"]:.3f}' for c in REGIONS)+' |')
    lines += ['', '**Limits of this result:** the scorer resizes each complete image to 224×224, then reads 5×5 windows from a 16×16 DINO grid. A full window spans about 31% of each image dimension. It was trained for retrieval; these scores have not been validated as tiny-detail accuracy. Molmo location errors can affect them. Buttons have 8 examples and pockets 3. Lapels, buttonholes and zippers have no eligible GT detections, so there are no scores for them.', '',
              '## GT and generated details', '',
              'These are the same seven cases selected for the original gallery, one per measured category, using the largest spread between model scores. This selection emphasizes disagreement and is not a representative random sample. Each sheet shows the garment reference, GT, the scorer’s input resolution, then GT and all eight generated crops at identical coordinates. The displayed score aggregates up to four points; the crop shows the first. No visual winner is assigned.', '']
    for r in selections:
        lines += ['### '+r['concept'].title(), '',
                  f'Case `{r["case"]}`. [Open full-resolution sheet](comparison-images/{r["image"]}).', '',
                  *(['**Label issue:** Molmo called this a collar; the visible feature is a neckline. We retain the original label and scores so the existing benchmark remains auditable.', ''] if r['concept']=='collar' else []),
                  f'![{r["concept"].title()} comparison: GT and all eight VTON models](comparison-images/{r["image"]})', '']
    (site/'README.md').write_text('\n'.join(lines))
    # Keep the already-published webpage consistent with the new primary figure.
    index = site/'index.html'
    text = index.read_text()
    new_section = '<section class="chart-card" aria-label="Three graph options"><p class="chart-note"><a href="detail_heatmap.png">Heatmap PNG</a> · <a href="detail_bars.png">Bar chart PNG</a> · <a href="region_profiles.png">Original line chart PNG</a> · <a href="https://github.com/LinkaiShao/qwen_vton/blob/master/docs/region-benchmark/README.md">All graphs and GT/model images directly on GitHub</a></p><h2 style="padding:0 18px">Heatmap</h2><img src="detail_heatmap.png" alt="Heatmap of mean distances" style="display:block;width:100%;height:auto"><h2 style="padding:0 18px">Bar chart</h2><img src="detail_bars.png" alt="Grouped model bars with confidence intervals" style="display:block;width:100%;height:auto"><h2 style="padding:0 18px">Original line chart</h2><img src="region_profiles.png" alt="Original eight-model line chart" style="display:block;width:100%;height:auto"></section>'
    text, count = re.subn(r'<section class="chart-card"[^>]*>.*?</section>', lambda _: new_section, text, count=1, flags=re.S)
    assert count == 1
    text = re.sub(r'<script>.*?</script>', '', text, flags=re.S)
    text = text.replace('eight VTON models as separate curves', 'eight VTON models shown as a heatmap, grouped bars and lines')
    text = text.replace('<strong>Each curve:</strong>', '<strong>Each color:</strong>')
    text = text.replace('They are omitted rather than plotted as zero.', 'There are no scores for these categories.')
    index.write_text(text)
    repo_readme = site.parents[1]/'README.md'
    text = repo_readme.read_text()
    message = '**[Eight-model VTON detail comparison — graph and GT/model image sheets](docs/region-benchmark/README.md)** · [Architecture report](docs/region-benchmark/architecture-report.md) · [Model training table](docs/region-benchmark/model-training.md)\n\n'
    if 'Eight-model VTON detail comparison — graph and GT/model image sheets' not in text:
        first, rest = text.split('\n', 1)
        repo_readme.write_text(first+'\n\n'+message+rest.lstrip('\n'))
    assert source_hashes == {name: digest(report/name) for name in source_hashes}
    print(json.dumps(dict(images=len(selections), cases=[r['case'] for r in selections], scores_changed=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--site', type=Path, required=True)
    args = p.parse_args()
    export(args.job, args.site)
