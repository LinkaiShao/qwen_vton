"""Publish every full-sample evaluation image and its identical fixed ROI crops."""
import argparse,json,shutil
from pathlib import Path
from PIL import Image
from build_report import crop,CONCEPTS,LABELS,bootstrap_delta

def build(job):
    source=job/'full_samples_v2';result=json.loads((source/'RESULT.json').read_text())
    manifest=json.loads((job/'manifest.json').read_text())
    dest=job/'site/full-samples';assets=dest/'assets';assets.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/'RESULT.json',dest/'RESULT.json')
    arms=['baseline','denoise_spatial_control','denoise_spatial_dino']
    names=['Initial LeFFA','Localization control','Localization + DINO']
    lookup={(r['arm'],r['id'],r['concept']):r['fixed_roi_dino_distance'] for r in result['records']}
    stats=bootstrap_delta([dict(r,t_fraction=1) for r in result['records'] if r['arm']=='denoise_spatial_control'],
                          [dict(r,t_fraction=1) for r in result['records'] if r['arm']=='denoise_spatial_dino'],
                          'fixed_roi_dino_distance')
    (dest/'paired_image_statistics.json').write_text(json.dumps(stats,indent=2))
    cards=[]
    for row in manifest['records']:
        if row['split']!='validation':continue
        iid=row['id'];folder=assets/iid;folder.mkdir(exist_ok=True)
        images=[Image.open(job/'data'/iid/'target.png').convert('RGB')]+[Image.open(source/arm/(iid+'.png')).convert('RGB') for arm in arms]
        labels=['Real target']+names
        for j,image in enumerate(images):image.save(folder/f'{j}.png')
        full=''.join(f'<figure><img loading="lazy" src="assets/{iid}/{j}.png"><figcaption>{label}</figcaption></figure>' for j,label in enumerate(labels))
        details=[]
        for k,(concept,label) in enumerate(zip(CONCEPTS,LABELS)):
            if not row['visible'][k]:continue
            boxes=row['boxes'][k]
            for j,image in enumerate(images):crop(image,boxes,folder/f'{concept}_{j}.png')
            captions=['Real target crop']+[f'{name}<br>DINO distance {lookup[(arm,iid,concept)]:.4f}' for arm,name in zip(arms,names)]
            crops=''.join(f'<figure><img loading="lazy" src="assets/{iid}/{concept}_{j}.png"><figcaption>{caption}</figcaption></figure>' for j,caption in enumerate(captions))
            details.append(f'<details><summary>{label}: same crop in every image</summary><div class="grid">{crops}</div></details>')
        cards.append(f'<article><h2>{iid}</h2><div class="grid">{full}</div>{"".join(details)}</article>')
    rows=[]
    for arm,name in zip(arms,names):
        values=[]
        for concept in CONCEPTS:
            v=[r['fixed_roi_dino_distance'] for r in result['records'] if r['arm']==arm and r['concept']==concept]
            values.append(sum(v)/len(v))
        rows.append(f'<tr><td>{name}</td>'+''.join(f'<td>{v:.4f}</td>' for v in values)+f'<td>{result["mean_fixed_roi_dino_distance"][arm]:.4f}</td></tr>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LeFFA · full-sample transfer check</title><style>*{box-sizing:border-box}body{background:#edf2f4;color:#233641;font:16px/1.55 system-ui;margin:0}main{max-width:1300px;padding:30px 24px;margin:auto}h1{font-size:36px;line-height:1.15}h2{font-size:21px}a{color:#087786}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}figure{margin:0}img{width:100%;display:block}figcaption{font-size:13px;margin-top:7px}article{background:white;border-radius:10px;padding:22px;margin:26px 0}details{padding-top:16px;border-top:1px solid #dae4e7;margin-top:18px}summary{cursor:pointer;font-weight:700;margin-bottom:12px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{text-align:left;padding:12px;border-bottom:1px solid #dae4e7}th{background:#dce8ed}.scroll{overflow-x:auto}.notice{background:#fff8ea;padding:20px;border-left:4px solid #c58b2d}@media(max-width:650px){main{padding:22px 12px}.grid{grid-template-columns:repeat(2,1fr)}article{padding:14px}}</style><main><a href="../">← Experiment overview</a><h1>Does the loss improvement transfer to full samples?</h1><div class="notice">These images are generated with the official LeFFA pipeline from random noise: 20 DDPM steps, guidance 2.5, 512×384, no final RGB repaint. All three variants use matched per-image seeds. <strong>These are separate post-training checks: no full sampling was used to compute a training loss.</strong></div><p>Every one of the eight development images is shown. The same automatic detail boxes score every variant. This remains a small development-cohort result, not an independent benchmark.</p><div class="scroll"><table><tr><th>Fixed-crop DINO distance ↓</th><th>Collar</th><th>Left sleeve</th><th>Right sleeve</th><th>Overall</th></tr>'''+''.join(rows)+'''</table></div><p>A lower feature distance is not by itself evidence that a visible defect was fixed; inspect the paired crops below. <a href="RESULT.json">Raw per-component scores and seeds</a> · <a href="../followup/">Training-step heatmaps and reconstruction results</a></p>'''+''.join(cards)+'</main></html>'
    interval=stats['bootstrap_95pct_image_interval']
    page=page.replace('<h1>Does the loss improvement transfer to full samples?</h1>',
                      '<h1>Does the loss improvement transfer to full samples?</h1>' +
                      f'<p>Paired image mean DINO delta (joint − control): {stats["mean_paired_image_delta"]:+.5f}. Descriptive image-bootstrap 95% interval: [{interval[0]:+.5f}, {interval[1]:+.5f}], n=8. <a href="paired_image_statistics.json">Statistics</a>.</p>')
    (dest/'index.html').write_text(page);print(dest/'index.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);build(p.parse_args().job)
