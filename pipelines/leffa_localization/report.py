"""Scientific audit report: explicit mask failures, raw predictions and interventions."""
import html
import shutil
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from common import *
from annotate import instance_masks

STYLE='''*{box-sizing:border-box}body{margin:0;background:#edf1f4;color:#162b35;font:15px/1.5 system-ui}main{max-width:1550px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:20px}.status,article{background:white;padding:18px;margin:18px 0;border-radius:8px}.status{border-left:5px solid #ba681e}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}.trace{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}figure{margin:0}img{display:block;width:100%}figcaption{font-size:12px}.overlay{position:relative}.overlay canvas{position:absolute;inset:0;width:100%;height:100%}label{margin-right:18px}select{padding:6px}a{color:#00637b}table{border-collapse:collapse}td,th{padding:7px;text-align:left;border-bottom:1px solid #ccc}.scroll{overflow:auto}.failure{color:#a1271d}@media(max-width:800px){.grid,.trace{grid-template-columns:repeat(2,minmax(0,1fr))}main{padding:12px}}'''

def build(root):
    dest=root/'site';dest.mkdir(exist_ok=True);assets=dest/'assets';assets.mkdir(exist_ok=True)
    st=read(root/'STATE.json') if (root/'STATE.json').exists() else {'stage':'preparing'}
    result=read(root/'RESULT.json') if (root/'RESULT.json').exists() else None
    selection=read(root/'SELECTION.json') if (root/'SELECTION.json').exists() else None
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LeFFA detail localization audit</title><style>'+STYLE+'</style><main>',
           '<a href="../">Previous experiment</a><h1>Where does LeFFA generate this detail?</h1><div class="status"><strong>'+html.escape(result['status'] if result else st['stage'])+'</strong>',
           '<p>Frozen original LeFFA · RTX 5090 · no DINO appearance loss · no training updates.</p>']
    for file in ['ANNOTATION_PROGRESS.json','MASK_PROGRESS.json','PROBE_PROGRESS.json']:
        if (root/file).exists():
            p=read(root/file);parts.append('<p>'+html.escape(' · '.join(str(p[k]) for k in ['view','role','done','total','key','status'] if k in p))+'</p>')
    if st.get('error'):parts.append('<p class="failure">'+html.escape(st['error'])+'</p>')
    if (root/'SMOKE.json').exists():
        smoke=read(root/'SMOKE.json');verified=read(root/'VERIFIED.json')
        parts.append(f'<p>Real GPU smoke check: {len(smoke["records"])} case/timestep probes, maximum forward difference with tracing {verified["max_output_difference"]}, peak allocated GPU memory {smoke["peak_gpu_gib"]:.2f} GiB. These checks verify the mechanism, not localization accuracy.</p>')
    parts.append('</div><p>Source detail masks come from agreement between two frozen SAM3 prompts. Qwen only names the garment. Orange marks are computed from the generator’s own Q/K attention. Red maps show where intervening on those source values actually changed this same single-pass prediction. No target annotation moves the predicted mark.</p><p>The finest native grid has 8-pixel cells; earlier layers have 16-pixel cells and are interpolated for display. Source/target annotations are automatic and can be wrong; a heatmap alone is not proof. Normalized display brightness does not imply confidence. Missing detections are uncertain, not proof of absence.</p>')
    if result:
        shutil.copy2(root/'RESULT.json',dest/'RESULT.json');parts.append('<p><a href="RESULT.json">Full gate results and causal confidence intervals</a></p><p>Raw accuracy is measured before confidence rejection. Accepted coverage shows how often a mark meets the frozen development threshold. A correct peak with diffuse mass still fails the detail-localization gate.</p><div class="scroll"><table><tr><th>Detail</th><th>Noise band</th><th>Raw samples</th><th>Raw peak hit</th><th>Raw inside mass</th><th>Raw generated-image peak hit</th><th>Accepted coverage</th><th>Pass</th></tr>')
        for name,c in result['components'].items():
            for b,m in c['by_noise_bin'].items():parts.append(f'<tr><td>{name}</td><td>{int(b)*200}–{int(b)*200+199}</td><td>{m["unfiltered_count"]}</td><td>{m["unfiltered_peak_hit"]:.1%}</td><td>{m["unfiltered_inside_mass"]:.1%}</td><td>{m["unfiltered_actual_peak_hit"]:.1%} (n={m["unfiltered_actual_count"]})</td><td>{m["coverage"]:.1%}</td><td>{m["passed"]}</td></tr>')
        parts.append('</table></div>')
    rows=manifest(root)
    selected=[r for r in rows if r['role']=='regression']+[r for r in rows if r['role']=='heldout'][:8]+[r for r in rows if r['role']=='calibration'][:4]
    parts.append('<p>Displayed cases: all eight known failures, the first eight preregistered fresh test cases, and four development cases. The selection is fixed before scores.</p>')
    for row in selected:
        key=row['key'];f=root/'data'/key;out=assets/key;out.mkdir(exist_ok=True)
        meta=read(f/'IDENTITY.json') if (f/'IDENTITY.json').exists() else None
        parts.append('<article data-key="'+key+'"><h2>'+key+' · '+row['role']+'</h2><div class="grid">')
        for name,label in [('garment','Flatlay reference'),('target','Real worn image'),('old_identity','Old garment mask'),('identity','Corrected garment identity'),('protected','Protected body / other garments'),('inpaint','New inpainting mask')]:
            p=f/(name+'.png')
            if not p.exists():continue
            shutil.copy2(p,out/p.name);parts.append(f'<figure><img loading="lazy" src="assets/{key}/{p.name}" alt="{label}"><figcaption>{label}</figcaption></figure>')
        parts.append('</div>')
        if meta:
            if meta['status']=='failed':parts.append('<p class="failure">Identity failed: '+html.escape(meta['error'])+'. No old-mask fallback.</p>')
            else:parts.append(f'<p>Protected pixels erased by inpainting: old {meta["old_inpaint_protected_pixels"]:,}; new {meta["new_inpaint_protected_pixels"]:,}.</p>')
        data={};frames=[];source_masks=None;instance_names=[]
        if (f/'parts.npz').exists():
            source_masks=np.load(f/'parts.npz')['source'];instances,instance_names=instance_masks(source_masks)
            source_masks=np.concatenate([source_masks,instances]);original=np.asarray(Image.open(f/'garment.png').convert('RGB'))
            for j,p in enumerate([*PARTS,*instance_names]):
                overlay=original.copy();mask=source_masks[j];overlay[mask]=(overlay[mask]*.5+np.array([255,125,0])*.5).astype('uint8')
                Image.fromarray(overlay).save(out/('source_'+p+'.png'))
        for t in TIMESTEPS:
            npz=root/'probe'/key/f'{t}.npz';png=npz.with_suffix('.png');receipt=npz.with_suffix('.json')
            is_smoke=False
            if not npz.exists():npz=root/'smoke'/key/f'{t}.npz';png=npz.with_suffix('.png');is_smoke=True
            if not (npz.exists() and png.exists()):continue
            if not is_smoke and not receipt.exists():continue
            with np.load(npz) as pack:
                maps=pack['maps'];separate=pack['instance_maps'] if 'instance_maps' in pack else None
                names=pack['candidates'].tolist() if 'candidates' in pack else None
            if maps.ndim==4:
                maps=np.stack([maps[names.index(selection[p]['candidate'] if selection else 'all/mean'),j] for j,p in enumerate(PARTS)])
            if separate is not None and len(separate):maps=np.concatenate([maps,separate])
            shutil.copy2(png,out/f'{t}.png');data[str(t)]={};frames.append(t)
            available=[*PARTS,*instance_names][:len(maps)]
            for j,p in enumerate(available):
                a=maps[j];ca=root/'causal'/key/f'{t}_{p}.npz';effect=np.load(ca)['effect'] if ca.exists() else None
                parent=p.split('_source_instance_')[0];source=read(root/'annotations/source'/(key+'.json'))['parts'][parent]
                data[str(t)][p]={'attention':np.round(a/max(1e-12,float(a.max()))*255).astype(np.uint8).flatten().tolist(),
                    'absolute_mass':float(a.sum()),'source_image':'source_'+p+'.png','source_boxes':[],'source_status':source['status'],
                    'verified':bool(read(receipt)['scores'].get(p,{}).get('confident',False)) if not is_smoke and row['role']!='calibration' and p in PARTS else False,
                    'readout':selection[parent]['candidate'] if selection else 'all/mean (unselected smoke diagnostic)',
                    'effect':np.round(effect/max(1e-12,float(effect.max()))*255).astype(np.uint8).flatten().tolist() if effect is not None else None}
                changed=root/'causal'/key/f'{t}_{p}.png'
                if changed.exists():shutil.copy2(changed,out/changed.name)
        if frames:
            (out/'traces.json').write_text(json.dumps(data,separators=(',',':')))
            parts.append('<p><label>Detail <select class="part">'+''.join(f'<option value="{p}">{p.replace("_"," ")}</option>' for p in data[str(frames[0])])+'</select></label><label>Timestep <select class="time">'+''.join(f'<option>{t}</option>' for t in frames)+'</select></label></p><div class="trace">')
            for label,role in [('Source detail (automatic annotation)','source'),('Raw single-pass prediction','raw'),('Generator attention — orange','attention'),('Measured output change — red','effect')]:
                src='garment.png' if role=='source' else str(frames[0])+'.png'
                parts.append(f'<figure><div class="overlay"><img loading="lazy" data-role="{role}" src="assets/{key}/{src}"><canvas data-role="{role}" width="384" height="512"></canvas></div><figcaption>{label}</figcaption></figure>')
            parts.append('</div><p class="trace-status"></p>')
        for name in ['old','corrected']:
            src=root/'mask_comparison'/key/(name+'.png')
            if src.exists():shutil.copy2(src,out/('mask_'+name+'.png'))
        if (out/'mask_old.png').exists():parts.append(f'<details><summary>Frozen model: old versus corrected mask at timestep 499</summary><div class="trace"><figure><img loading="lazy" src="assets/{key}/mask_old.png"><figcaption>Old mask</figcaption></figure><figure><img loading="lazy" src="assets/{key}/mask_corrected.png"><figcaption>Corrected mask · same checkpoint and noise</figcaption></figure></div></details>')
        parts.append('</article>')
    parts.append('''<script>
for(const el of document.querySelectorAll('article')){const part=el.querySelector('.part'),time=el.querySelector('.time');if(!part)continue;let data;
async function update(){if(!data)data=await(await fetch('assets/'+el.dataset.key+'/traces.json')).json();const d=data[time.value][part.value],base='assets/'+el.dataset.key+'/';
for(const im of el.querySelectorAll('img[data-role]'))im.src=base+(im.dataset.role==='source'?d.source_image:time.value+'.png');
for(const canvas of el.querySelectorAll('canvas')){const ctx=canvas.getContext('2d');ctx.clearRect(0,0,384,512);const role=canvas.dataset.role;if(role==='source'){ctx.strokeStyle='#e87714';ctx.lineWidth=3;for(const b of d.source_boxes)ctx.strokeRect(b[0]*384,b[1]*512,(b[2]-b[0])*384,(b[3]-b[1])*512)}else if(role==='attention'||role==='effect'){const v=role==='attention'?d.attention:d.effect;if(!v)continue;const tmp=document.createElement('canvas');tmp.width=48;tmp.height=64;const c=tmp.getContext('2d'),im=c.createImageData(48,64);for(let i=0;i<v.length;i++){im.data[4*i]=255;im.data[4*i+1]=role==='attention'?140:20;im.data[4*i+2]=0;im.data[4*i+3]=Math.round(v[i]*.65)}c.putImageData(im,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(tmp,0,0,384,512)}}
el.querySelector('.trace-status').textContent='Source status: '+d.source_status+'. '+(d.verified?'Passes development confidence threshold.':'UNVERIFIED location — no passing calibrated confidence.')+' Readout: '+d.readout+'. Absolute attention mass: '+d.absolute_mass.toFixed(4)+(d.effect?' · Intervention measured.':' · No intervention evidence yet for this mark.');}
part.addEventListener('change',update);time.addEventListener('change',update);update();}
</script></main></html>''')
    (dest/'index.html').write_text(''.join(parts));print('REPORT',dest/'index.html',flush=True)

if __name__=='__main__':a=arguments(__doc__).parse_args();build(a.root)
