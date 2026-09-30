"""Render the published location/score/model comparison from audited means."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REGIONS = ['collar', 'shoulder', 'sleeve', 'cuff', 'button', 'pocket', 'logo']
LABELS = ['Collar', 'Shoulder', 'Sleeve', 'Cuff', 'Buttons', 'Pockets', 'Logo']
MODELS = {
    'coral': ('CORAL', '#0072B2', 'o'),
    'deco': ('DeCo-VTON', '#D55E00', 's'),
    'fastfit': ('FastFit', '#7C3AED', '^'),
    'leffa': ('Leffa', '#00845A', 'D'),
    'catflux': ('CatVTON-FLUX', '#A23A72', 'P'),
    'catvton': ('CatVTON', '#525F70', 'X'),
    'idm': ('IDM-VTON', '#BA7A00', 'v'),
    'ootd': ('OOTDiffusion', '#009AB2', '*'),
}


def publish(report: Path, site: Path):
    summary = json.loads((report / 'summary.json').read_text())
    assert summary['common_n'] == 128 and summary['models'] == list(MODELS)
    cells = {(c['model'], c['concept']): c for c in summary['cells']}
    counts = [cells['coral', region]['n'] for region in REGIONS]
    for region, n in zip(REGIONS, counts):
        assert n > 0
        assert all(cells[m, region]['n'] == n for m in MODELS)
    absent = [c['concept'] for c in summary['cells'] if c['model'] == 'coral' and c['n'] == 0]
    values = [cells[m, r]['mean'] for m in MODELS for r in REGIONS]
    lo = max(0, math.floor((min(values) - .04) * 20) / 20)
    hi = min(2, math.ceil((max(values) + .04) * 20) / 20)
    plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans',
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#172238',
                         'text.color': '#172238', 'xtick.color': '#334155',
                         'ytick.color': '#334155'})
    fig, ax = plt.subplots(figsize=(14, 6.8))
    fig.subplots_adjust(left=.085, right=.975, top=.91, bottom=.29)
    x = np.arange(len(REGIONS))
    for i, n in enumerate(counts):
        if n < 10:
            ax.axvspan(i - .46, i + .46, color='#fff4db', zorder=0)
    for model, (name, color, marker) in MODELS.items():
        y = [cells[model, region]['mean'] for region in REGIONS]
        line, = ax.plot(x, y, label=name, color=color, marker=marker,
                        linewidth=2.0, markersize=7.5, markeredgecolor='white',
                        markeredgewidth=.6, alpha=.95)
        line.set_gid('series-' + model)
    ax.set_xlim(-.25, len(REGIONS) - .75)
    ax.set_ylim(lo, hi)
    ax.set_xticks(x, [f'{label}\nn = {n}' for label, n in zip(LABELS, counts)])
    ax.set_yticks(np.arange(lo, hi + .001, .05))
    ax.set_ylabel('Regional score — mean expert L2 distance ↓', labelpad=12)
    ax.set_xlabel('Garment location', labelpad=12)
    ax.grid(axis='y', color='#dbe2eb', linewidth=.7)
    ax.grid(axis='x', color='#edf0f5', linewidth=.65)
    ax.set_axisbelow(True)
    ax.set_title('One curve per VTON model · lower scores are better', loc='left', pad=14, fontsize=16)
    legend = ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, -.24),
                       frameon=False, handlelength=2.7, columnspacing=2.4, fontsize=11)
    legend.set_gid('static-legend')
    fig.text(.085, .018, f'128 matched test cases. Shaded locations have fewer than 10 cases. '
             f'Axis shown: {lo:.2f}–{hi:.2f}; possible score range: 0–2.', fontsize=10, color='#475569')
    site.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'svg']:
        fig.savefig(site / ('region_profiles.' + extension), dpi=180, facecolor='white')
    plt.close(fig)

    ns = {'s': 'http://www.w3.org/2000/svg'}
    ET.register_namespace('', ns['s'])
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree = ET.parse(site / 'region_profiles.svg')
    svg = tree.getroot()
    svg.set('role', 'img')
    svg.set('aria-labelledby', 'plot-title plot-description')
    ET.SubElement(svg, '{'+ns['s']+'}title', id='plot-title').text = 'Garment locations versus regional scores for eight VTON models'
    ET.SubElement(svg, '{'+ns['s']+'}desc', id='plot-description').text = (
        'The horizontal axis is garment location. The vertical axis is mean expert L2 distance, '
        'where lower is better. Each colored curve is one model. The exact values and confidence '
        'intervals are available in the table below the graph.')
    for model, (name, _, _) in MODELS.items():
        group = svg.find(f'.//s:g[@id="series-{model}"]', ns)
        assert group is not None
        title = ET.Element('{'+ns['s']+'}title')
        title.text = name + ': ' + '; '.join(f'{label} {cells[model, region]["mean"]:.3f}' for region, label in zip(REGIONS, LABELS))
        group.insert(0, title)
        group.set('class', 'model-series')
    inline = ET.tostring(svg, encoding='unicode')
    buttons = ''.join(f'<button type="button" class="model" data-model="{model}" aria-pressed="false" '
                      f'style="--color:{color}"><span class="swatch"></span>{html.escape(name)}</button>'
                      for model, (name, color, _) in MODELS.items())
    table = ['<table><caption>Mean distance and 95% image-bootstrap interval</caption><thead><tr><th>Model</th>']
    for label, n in zip(LABELS, counts):
        table.append(f'<th>{label}<br><span class="count">n = {n}</span></th>')
    table.append('</tr></thead><tbody>')
    for model, (name, _, _) in MODELS.items():
        table.append(f'<tr><th scope="row">{name}</th>')
        for region in REGIONS:
            c = cells[model, region]
            table.append(f'<td><strong>{c["mean"]:.3f}</strong><br><span class="interval">{c["low"]:.3f}–{c["high"]:.3f}</span></td>')
        table.append('</tr>')
    table.append('</tbody></table>')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Garment locations on the x-axis, regional scores on the y-axis, and eight VTON models as separate curves.">
<title>VTON scores by garment location</title>
<style>
*{box-sizing:border-box}html{color-scheme:light}body{margin:0;background:#f4f6fa;color:#172238;font:16px/1.55 system-ui,-apple-system,sans-serif}main{max-width:1500px;margin:auto;padding:24px clamp(14px,3vw,40px) 40px}header{display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap}h1{font-size:clamp(25px,3vw,34px);line-height:1.2;margin:0 0 8px}p{margin:8px 0;color:#49576b}a{color:#1558af;text-underline-offset:3px}a:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #1558af;outline-offset:4px}.primary{padding:10px 16px;background:#1558af;color:white;border-radius:7px;text-decoration:none;font-weight:600}.key{margin:18px 0 14px;color:#243247}.chart-card{background:white;border:1px solid #d6dfe9;border-radius:8px}.toolbar{padding:14px 16px 10px;border-bottom:1px solid #e2e8f0}.toolbar p{margin:0 0 10px;font-size:14px}.models{display:flex;flex-wrap:wrap;gap:8px}button{font:600 14px/1.3 system-ui;background:white;border:1px solid #cfd8e5;border-radius:6px;padding:8px 11px;color:#1e293b;cursor:pointer}.model{display:flex;gap:8px;align-items:center}.model[aria-pressed=true]{border-color:var(--color);box-shadow:inset 0 0 0 1px var(--color);background:#f2f5fa}.swatch{width:20px;height:4px;background:var(--color);display:inline-block}.chart{overflow-x:auto;padding:8px}.chart svg{display:block;width:100%;height:auto;min-width:950px}.model-series{transition:opacity .16s}.chart-note{padding:0 18px 14px;font-size:14px;color:#475569}.status{min-height:22px;font-size:14px;color:#334155;margin:10px 0 0!important}details{margin-top:22px;background:white;border:1px solid #d6dfe9;border-radius:8px;padding:15px 18px}summary{font-weight:600;cursor:pointer}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap;margin-top:18px}caption{text-align:left;color:#475569;padding-bottom:10px}th,td{text-align:left;padding:11px 14px;border-bottom:1px solid #e2e8f0}.count,.interval{font-weight:400;color:#5a687c}footer{margin-top:22px}nav{display:flex;flex-wrap:wrap;gap:12px 24px}.meta{font-size:14px}.secondary-plot{display:block;width:100%;height:auto;min-width:1000px;margin-top:14px}
</style></head><body><main>
<header><div><h1>VTON scores by garment location</h1><p>8 models · 128 matched VITON-HD cases · 1,024 generated images</p></div><a class="primary" href="report.html">View GT &amp; model outputs</a></header>
<p class="key"><strong>X:</strong> garment location &nbsp; <strong>Y:</strong> regional distance score, lower is better &nbsp; <strong>Each curve:</strong> one VTON model</p>
<section class="chart-card" aria-label="Model performance across garment locations"><div class="toolbar"><p>Select a model to highlight its curve.</p><div class="models">''' + buttons + '''<button type="button" id="show-all">Show all models</button></div><p id="selection" class="status" role="status">Showing all eight models.</p></div><div class="chart" tabindex="0">''' + inline + '''</div><p class="chart-note">Compare models <strong>within each location</strong>. The lines connect category means; they do not imply a spatial path. The shaded button and pocket locations have only 8 and 3 cases. <a href="region_profiles.png">Open full-size graph</a> · <a href="region_profiles.svg" download>Download SVG</a></p></section>
<p class="meta">Not measured: ''' + ', '.join(html.escape(a) for a in absent) + ''' — no eligible GT examples. They are omitted rather than plotted as zero.</p>
<details><summary>Exact scores, sample counts and confidence intervals</summary><div class="table-wrap">''' + ''.join(table) + '''</div></details>
<details><summary>All 95% confidence intervals</summary><div class="table-wrap"><img class="secondary-plot" loading="lazy" src="region_scores.svg" alt="Grouped regional distances with 95 percent image-bootstrap intervals for all eight models."></div><p class="meta">Intervals resample the matched images 2,000 times. Small differences between means should be assessed with their uncertainty.</p></details>
<footer><nav><a href="report.html">Full comparison &amp; Molmo locations</a><a href="scores.csv" download>Download per-image regional scores</a><a href="summary.json" download>Download summary</a><a href="../molmo-20260922/">Previous training comparison</a></nav><p class="meta">Molmo identifies parts; frozen DINO experts score each model at the same GT locations. Scores are feature distances, not accuracy percentages. The underlying 128-case results are unchanged.</p></footer>
<script>
const modelNames=''' + json.dumps({k: v[0] for k, v in MODELS.items()}) + ''';
let active=null;
function highlight(model){
 active=model===active?null:model;
 for(const key of Object.keys(modelNames)){
  document.getElementById('series-'+key).style.opacity=active&&active!==key?'0.12':'1';
  document.querySelector('[data-model="'+key+'"]').setAttribute('aria-pressed',String(active===key));
 }
 document.getElementById('selection').textContent=active?'Highlighted: '+modelNames[active]+'. Other models remain visible for comparison.':'Showing all eight models.';
}
document.querySelectorAll('[data-model]').forEach(button=>button.addEventListener('click',()=>highlight(button.dataset.model)));
document.getElementById('show-all').addEventListener('click',()=>{active=null;highlight(null)});
</script></main></body></html>'''
    (site / 'index.html').write_text(page)
    print(json.dumps(dict(regions=REGIONS, counts=counts, models=list(MODELS), axis=[lo, hi], page=str(site/'index.html'))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--site', required=True, type=Path)
    a = p.parse_args()
    publish(a.report, a.site)
