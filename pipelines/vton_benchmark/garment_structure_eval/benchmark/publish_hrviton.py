"""Publish the explicit-flow baseline beside the immutable eight-model results."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .common import atomic, digest, read, rows
from .publish_profiles import MODELS, REGIONS, LABELS
from .export_comparison_images import font, put_image, window

NAMES={k:('FastFit-SR' if k=='fastfit' else v[0]) for k,v in MODELS.items()}
NAMES.update(hrviton='HR-VITON · shared mask',hrviton_native='HR-VITON · native mask')
PLOT_MODELS=list(MODELS)+['hrviton']


def analyze(a):
    by={}
    for key in NAMES:
        root=a.source/'collected' if key in MODELS else a.job
        for p in (root/'scores'/key).glob('*.json'):
            r=read(p);by[key,r['id']]=r
    ids=[r['id'] for r in rows(a.job)]
    assert len(ids)==128
    for m in NAMES:assert {i for k,i in by if k==m}==set(ids)
    cells=[];deltas=[]
    for concept in REGIONS:
        eligible=[];values=[]
        for iid in ids:
            parts=[next(p for p in by[m,iid]['parts'] if p['concept']==concept) for m in NAMES]
            assert len({p['gt_present'] for p in parts})==1
            assert all(p['gt_points']==parts[0]['gt_points'] for p in parts)
            if parts[0]['gt_present']:
                eligible.append(iid);values.append([p['fixed_distance'] for p in parts])
        matrix=np.asarray(values);assert np.isfinite(matrix).all()
        rng=np.random.default_rng(20260924)
        indices=rng.integers(0,len(eligible),size=(2000,len(eligible)))
        boot=matrix[indices].mean(1);ci=np.percentile(boot,[2.5,97.5],axis=0)
        for j,m in enumerate(NAMES):
            cells.append(dict(model=m,concept=concept,n=len(eligible),mean=float(matrix[:,j].mean()),
                low=float(ci[0,j]),high=float(ci[1,j])))
        for reference in ['leffa','ootd','hrviton_native']:
            i=list(NAMES).index('hrviton');j=list(NAMES).index(reference)
            d=matrix[:,i]-matrix[:,j];interval=np.percentile(boot[:,i]-boot[:,j],[2.5,97.5])
            deltas.append(dict(concept=concept,reference=reference,n=len(d),delta=float(d.mean()),
                low=float(interval[0]),high=float(interval[1]),hrviton_lower_count=int((d<0).sum())))
    old={(c['model'],c['concept']):c for c in read(a.source/'collected/report/summary.json')['cells']}
    for c in cells:
        if c['model'] in MODELS:
            prev=old[c['model'],c['concept']]
            assert prev['n']==c['n'] and abs(prev['mean']-c['mean'])<1e-12
    validation=read(a.job/'scorer_validation.json');assert validation['passed']
    inference=read(a.job/'inference_status.json');assert inference['complete']
    receipts=[read(a.job/'inference'/f'{iid}.json') for iid in ids]
    for r in receipts:
        for name,expected in r['artifacts'].items():assert digest(a.job/name)==expected
    timing=dict(gpu=inference['gpu'],warp_median_s=float(np.median([r['warp_seconds'] for r in receipts])),
        synthesis_median_s=float(np.median([r['generator_seconds']['hrviton'] for r in receipts])),
        total_median_s=float(np.median([r['warp_seconds']+r['generator_seconds']['hrviton'] for r in receipts])),
        peak_allocated_gib=max(r['max_allocated_bytes'] for r in receipts)/2**30)
    summary=dict(common_n=128,models=list(NAMES),plotted_models=PLOT_MODELS,cells=cells,
        paired_differences=deltas,timing=timing,scorer_validation=validation,
        score='Frozen regional expert L2 at original GT points; lower is better',
        source_summary_sha256=digest(a.source/'collected/report/summary.json'))
    return by,summary


def plots(summary,dest):
    cells={(c['model'],c['concept']):c for c in summary['cells']}
    counts=[cells['hrviton',c]['n'] for c in REGIONS]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    colors={k:v[1] for k,v in MODELS.items()};colors['hrviton']='#C41E3A'
    for kind in ['heatmap','bars','lines']:
        fig,ax=plt.subplots(figsize=(15,8.4));fig.subplots_adjust(left=.10,right=.98,top=.76,bottom=.21)
        if kind=='heatmap':
            fig.subplots_adjust(left=.23,right=.91,top=.79,bottom=.21)
            matrix=np.array([[cells[m,c]['mean'] for c in REGIONS] for m in PLOT_MODELS])
            im=ax.imshow(matrix,cmap='YlOrRd',vmin=.30,vmax=max(.55,float(matrix.max())),aspect='auto')
            ax.set_yticks(range(9),[NAMES[m] for m in PLOT_MODELS])
            for i in range(9):
                for j in range(7):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=14,
                    color='white' if matrix[i,j]>.5 else '#172238',weight='bold')
            fig.colorbar(im,ax=ax,fraction=.022,pad=.035,label='Mean feature distance ↓')
            ax.axhline(7.5,color='#C41E3A',linewidth=3)
        else:
            x=np.arange(7);width=.092
            for j,m in enumerate(PLOT_MODELS):
                vals=[cells[m,c] for c in REGIONS];y=np.array([v['mean'] for v in vals])
                if kind=='bars':
                    err=[[v['mean']-v['low'] for v in vals],[v['high']-v['mean'] for v in vals]]
                    ax.bar(x+(j-4)*width,y,width*.92,color=colors[m],label=NAMES[m],yerr=err,
                        error_kw=dict(elinewidth=.6,capsize=1.3,ecolor='#334155'))
                else:
                    ax.plot(x,y,color=colors[m],marker='o' if m=='hrviton' else MODELS[m][2],
                        linewidth=3.4 if m=='hrviton' else 1.5,markersize=8 if m=='hrviton' else 6,label=NAMES[m])
            ax.set_ylim(0,max(.8,max(cells[m,c]['high'] for m in PLOT_MODELS for c in REGIONS)+.04))
            if kind=='lines':ax.set_ylim(.28,max(.6,max(cells[m,c]['mean'] for m in PLOT_MODELS for c in REGIONS)+.045))
            ax.set_xlim(-.55,6.55);ax.set_ylabel('Mean regional feature distance to GT ↓')
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
            ax.legend(ncol=3,loc='lower left',bbox_to_anchor=(-.01,1.025),frameon=False,fontsize=11)
        ax.set_xticks(range(7),[f'{label}\nn = {n}' for label,n in zip(LABELS,counts)])
        ax.tick_params(axis='x',pad=10,length=0)
        fig.text(.045,.95,f'{kind.title()} · Does explicit garment warping help?',fontsize=23,weight='bold')
        fig.text(.045,.897,'HR-VITON added to the same 128-case comparison · original eight-model scores unchanged',fontsize=12)
        fig.text(.045,.095,'Lower feature distance is better. Primary HR-VITON uses the shared mask; native-mask sensitivity is tabulated separately.',fontsize=10.5)
        fig.text(.045,.052,'Scorer: full-image 224×224 DINO + learned regional experts. Not validated as tiny-detail accuracy. Buttons n=8; pockets n=3.',fontsize=10.5)
        for ext in ['png','svg']:fig.savefig(dest/f'{kind}.{ext}',dpi=160,facecolor='white')
        plt.close(fig)


def sheet(a,by,iid,dest,concept=None):
    record=next(r for r in rows(a.job) if r['id']==iid)
    parts={p['concept']:p for p in by['hrviton',iid]['parts']}
    if concept is None:
        concept=next((c for c in ['logo','collar','button','cuff','pocket','sleeve','shoulder'] if parts[c]['gt_present']),None)
    points=parts[concept]['gt_points'] if concept else []
    box=window(points[0],(768,1024)) if points else (0,0,768,1024)
    canvas=Image.new('RGB',(1456,782),'#f4f6fa');draw=ImageDraw.Draw(canvas)
    draw.text((20,14),f'{iid} · flatlay → pixel warp → final try-on',font=font(27,True),fill='#172238')
    draw.text((20,54),f"Detail below: {concept or 'full image'} · same GT crop in every person-space image",font=font(16),fill='#465569')
    paths=[('Flatlay reference',a.dataset/record['garment']),('GT',a.dataset/record['target']),
        ('Warp + occlusion mask',a.job/'warp'/f'{iid}.png'),('HR-VITON · shared mask',a.job/'predictions/hrviton'/f'{iid}.png'),
        ('HR-VITON · native mask',a.job/'predictions/hrviton_native'/f'{iid}.png'),
        ('LeFFA',a.source/'collected/predictions/leffa'/f'{iid}.png')]
    for j,(label,path) in enumerate(paths):
        x=20+j*240;im=Image.open(path).convert('RGB')
        draw.text((x,91),label,font=font(15,True),fill='#172238')
        marked=im.copy()
        if j>0 and points:ImageDraw.Draw(marked).rectangle(box,outline='#00a7cc',width=5)
        put_image(canvas,marked,(x,119,218,291))
        if j==0:
            draw.text((x,435),'Warp before occlusion',font=font(15,True),fill='#172238')
            put_image(canvas,Image.open(a.job/'raw_warp'/f'{iid}.png'),(x,467,218,246))
        else:
            draw.text((x,435),'GT detail' if j==1 else 'Same-location detail',font=font(15,True),fill='#172238')
            put_image(canvas,im.crop(box),(x,467,218,218))
        if j in [3,4,5] and concept:
            key={3:'hrviton',4:'hrviton_native',5:'leffa'}[j]
            d=next(p['fixed_distance'] for p in by[key,iid]['parts'] if p['concept']==concept)
            draw.text((x,705),f'Expert distance: {d:.3f}',font=font(15,True),fill='#172238')
    draw.text((20,750),'Cyan = first GT scoring window. Score aggregates up to 4 points; full image resized to 224×224. Crops show native pixels.',font=font(15),fill='#465569')
    if dest.suffix=='.jpg':canvas.save(dest,quality=90,subsampling=0)
    else:canvas.save(dest)
    return dict(id=iid,concept=concept,box=list(box),image=dest.name)


def publish(a):
    by,summary=analyze(a);a.site.mkdir(parents=True,exist_ok=True)
    plots(summary,a.site)
    cells={(c['model'],c['concept']):c for c in summary['cells']}
    atomic(a.site/'summary.json',summary)
    shutil.copy2(a.job/'provenance.json',a.site/'provenance.json')
    atomic(a.site/'scores.json',list(by.values()))
    with (a.site/'scores.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','id','concept','gt_present','fixed_distance'])
        for (m,iid),r in by.items():
            for p in r['parts']:w.writerow([m,iid,p['concept'],p['gt_present'],p['fixed_distance']])
    (a.site/'cases').mkdir(exist_ok=True);(a.site/'details').mkdir(exist_ok=True)
    cases=[sheet(a,by,r['id'],a.site/'cases'/f"{r['id']}.jpg") for r in rows(a.job)]
    selections=read(a.site.parent/'region-benchmark/comparison-images/manifest.json')['selections']
    details=[]
    for s in selections:
        details.append(sheet(a,by,s['case'],a.site/'details'/f"{s['concept']}.png",s['concept']))
    atomic(a.site/'gallery.json',dict(cases=cases,details=details,selection='All 128 cohort cases, original manifest order; detail sheets reuse the original seven selected cases'))
    means=['| Model | '+' | '.join(LABELS)+' |','|---|'+'---:|'*7]
    for m in NAMES:means.append('| '+NAMES[m]+' | '+' | '.join(f"{cells[m,c]['mean']:.3f}" for c in REGIONS)+' |')
    delta=['| Region | n | HR-VITON − LeFFA | Paired 95% interval |','|---|---:|---:|---|']
    for d in summary['paired_differences']:
        if d['reference']=='leffa':delta.append(f"| {d['concept'].title()} | {d['n']} | {d['delta']:+.3f} | {d['low']:+.3f} to {d['high']:+.3f} |")
    t=summary['timing']
    intro='HR-VITON explicitly predicts a flow field and resamples the garment pixels, then a SPADE image generator synthesizes the person wearing that warped garment. Official pretrained weights; no training performed.'
    assert all(cells['hrviton',c]['mean']>cells['leffa',c]['mean'] and cells['hrviton_native',c]['mean']>cells['leffa',c]['mean'] for c in REGIONS)
    result='Result: HR-VITON has higher (worse) mean expert distance than LeFFA in all seven measured regions. Native masking does not reverse this result. This older warping baseline is fast, but it does not improve the measured regional fidelity in this test.'
    protocol='Same 128 paired VITON-HD test cases and 1024×768 outputs. Primary HR-VITON uses the existing benchmark mask; native HR-VITON masking is a separate sensitivity check. Both retain required native parse-agnostic, cloth-mask and DensePose inputs. Original eight-model outputs and scores are unchanged.'
    limitations='Lower is closer in the learned feature space, not an accuracy percentage. Frozen DINO experts resize each full image to 224×224 and read 5×5 windows on a 16×16 grid. This retrieval-derived scorer is not validated as tiny-detail accuracy. We reuse the original Molmo GT points; new prediction localization is not evaluated. Buttons have 8 cases and pockets 3. No eligible GT detections for lapels, buttonholes or zippers. Automatic labels are retained from the original run; its collar example 14009_00 is visibly a neckline.'
    runtime=f"Measured on {t['gpu']}: median warp {t['warp_median_s']:.3f} s + image synthesis {t['synthesis_median_s']:.3f} s; median combined model time {t['total_median_s']:.3f} s/image. Excludes loading, preprocessing, PNG writes and scoring; batch one, FP32. Peak allocated memory {t['peak_allocated_gib']:.2f} GiB. H200 was unreachable; this run used the local GPU."
    links='[Live report](https://linkaishao.github.io/qwen_vton/warp-benchmark/) · [Original comparison](../region-benchmark/README.md) · [Official HR-VITON](https://github.com/sangyun884/HR-VITON) · [Run provenance](provenance.json) · [Raw scores](scores.csv)'
    md=['# Explicit garment warping versus the eight VTON models','',intro,'',links,'',result,'',protocol,'',
        '## Heatmap','','![Nine-model regional heatmap](heatmap.png)','','## Bar chart','','![Nine-model grouped bar chart](bars.png)','','## Line chart','','![Nine model series by garment location](lines.png)','','## Exact means','',*means,'',
        '## Paired comparison with LeFFA','','Negative differences favor HR-VITON. Intervals resample the same images for both models, 2,000 times.','',*delta,'',
        '## What the warp actually produced','','Each sheet shows flatlay, GT, warped garment, HR-VITON with shared mask, HR-VITON with native mask, and LeFFA. The seven cases are unchanged from the original gallery; the website includes all 128 cases.','']
    for d in details:md += [f"### {d['concept'].title()} · {d['id']}",'',f"![{d['concept']} warp and final result](details/{d['concept']}.png)",'']
    md += ['## Timing and validation','',runtime,'','The local scorer reproduced four previous LeFFA score records within 0.0001 and passed identical-image checks. Checkpoint loads were strict. All 128 inference artifacts and original image hashes were verified.','','## Interpretation limits','',limitations,'',
        'This comparison tests released systems and their conditioning setups. It does not isolate geometric warping as the cause of a score difference.']
    (a.site/'README.md').write_text('\n'.join(md)+'\n')
    write_html(a,summary,means,delta,intro+' '+result,protocol,limitations,runtime,cases,details)
    print(json.dumps(dict(site=str(a.site),cases=len(cases),details=len(details),summary=summary['timing'])),flush=True)


def write_html(a,summary,means,delta,intro,protocol,limitations,runtime,cases,details):
    def table(lines):
        cells=lambda s:s.strip('|').split('|')
        out='<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(c.strip())+'</th>' for c in cells(lines[0]))+'</tr></thead><tbody>'
        for line in lines[2:]:out+='<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in cells(line))+'</tr>'
        return out+'</tbody></table></div>'
    options=''.join(f'<option value="{c["id"]}">{i+1:03d} · {c["id"]} · {html.escape(c["concept"] or "full image")}</option>' for i,c in enumerate(cases))
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HR-VITON · Explicit garment warping comparison</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#172238;font:16px/1.6 system-ui}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:34px;line-height:1.2}h2{margin-top:32px}a{color:#1558af}nav{display:flex;gap:18px;flex-wrap:wrap}.card{background:white;border:1px solid #d6dfe9;border-radius:9px;padding:18px;margin:20px 0}.plot,.sheet{display:block;width:100%;height:auto}.table{overflow:auto}table{border-collapse:collapse;min-width:800px;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #dfe5ec}button,select{font:inherit;padding:8px 14px;border:1px solid #abb9cb;border-radius:5px;background:white;color:#172238}button{cursor:pointer}button[aria-pressed=true]{background:#172238;color:white}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #e49a11}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.muted{color:#526075}details{margin:18px 0}summary{cursor:pointer;font-weight:650}[hidden]{display:none!important}</style><main>
<h1>Does explicitly warping the garment help?</h1><nav><a href="../region-benchmark/">Original eight-model comparison</a><a href="https://github.com/LinkaiShao/qwen_vton/blob/master/docs/warp-benchmark/README.md">Read on GitHub</a><a href="scores.csv">Download scores</a><a href="provenance.json">Checkpoint and run provenance</a></nav>'''
    body+='<p>'+intro+'</p><p>'+protocol+'</p><p><strong>Lower regional feature distance is better.</strong> The primary graphs contain nine models. Native-mask HR-VITON is listed separately in the table.</p>'
    body+='<div class="card"><div class="controls" aria-label="Chart type">'+''.join(f'<button data-plot="{k}" aria-pressed="{str(k=="heatmap").lower()}">{k.title()}</button>' for k in ['heatmap','bars','lines'])+'</div>'
    for k in ['heatmap','bars','lines']:body+=f'<img class="plot" id="{k}" src="{k}.png" alt="Nine model regional {k}"'+(' hidden' if k!='heatmap' else '')+'>'
    body+='</div><h2>Exact scores</h2>'+table(means)+'<h2>HR-VITON versus LeFFA</h2><p>Negative differences favor HR-VITON. Paired 95% bootstrap intervals; 2,000 draws.</p>'+table(delta)
    body+='<h2>Inspect every case</h2><p>Flatlay → warped garment → final image, alongside GT and LeFFA. The selector includes all 128 cases in the original manifest order.</p><div class="card"><div class="controls"><button id="previous">Previous</button><label for="case">Case</label><select id="case">'+options+'</select><button id="next">Next</button><a id="open-case" href="cases/'+cases[0]['id']+'.jpg" target="_blank">Open full-size image</a></div><img id="case-image" class="sheet" src="cases/'+cases[0]['id']+'.jpg" alt="Flatlay, warp, HR-VITON, GT and LeFFA comparison"></div>'
    body+='<h2>Original detail examples</h2><p>Same seven cases as the earlier comparison. Cyan boxes mark shared GT scoring locations; the lower row shows larger crops.</p>'
    for d in details:body+=f'<details class="card"><summary>{html.escape(d["concept"].title())} · {d["id"]}</summary><a href="details/{d["concept"]}.png" target="_blank"><img loading="lazy" class="sheet" src="details/{d["concept"]}.png" alt="{html.escape(d["concept"])} comparison"></a></details>'
    body+='<h2>Measured runtime</h2><p>'+runtime+'</p><h2>What these scores can tell us</h2><p>'+limitations+'</p><p>Local scorer validation reproduced four original LeFFA records within 0.0001. The comparison is between complete released systems, not an experiment isolating warping alone.</p><p><a href="https://github.com/sangyun884/HR-VITON">HR-VITON official code and paper</a> · <a href="summary.json">Machine-readable summary and validation</a></p></main>'
    body+='''<script>document.querySelectorAll('[data-plot]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-plot]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));document.querySelectorAll('.plot').forEach(x=>x.hidden=x.id!==b.dataset.plot)}));const s=document.getElementById('case');function show(){const p='cases/'+s.value+'.jpg';document.getElementById('case-image').src=p;document.getElementById('open-case').href=p;document.getElementById('previous').disabled=s.selectedIndex===0;document.getElementById('next').disabled=s.selectedIndex===s.options.length-1;}s.addEventListener('change',show);document.getElementById('previous').onclick=()=>{s.selectedIndex--;show()};document.getElementById('next').onclick=()=>{s.selectedIndex++;show()};show();</script></html>'''
    (a.site/'index.html').write_text(body)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('/mnt/nvme0/vton_region_benchmark/20260925_eight_models_v2'))
    p.add_argument('--job',type=Path,default=Path('/mnt/nvme0/vton_region_benchmark/20260930_hrviton'))
    p.add_argument('--dataset',type=Path,default=Path('VITON-HD-dataset'))
    p.add_argument('--site',type=Path,default=Path('/mnt/nvme0/vton_region_benchmark/20260925_eight_models_v2/publish-github/docs/warp-benchmark'))
    publish(p.parse_args())
