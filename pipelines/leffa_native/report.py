"""Static comparison report; absent results are visibly pending, never invented."""
import argparse
import base64
import html
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
import numpy as np
from PIL import Image


STYLE='''*{box-sizing:border-box}body{margin:0;background:#edf1f3;color:#142a36;font:15px/1.5 system-ui}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:28px;margin:8px 0}h2{font-size:20px}p{max-width:1000px}a{color:#00647b}.status{padding:16px;background:white;border-left:5px solid #d39324}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}figure{margin:0;position:relative}img{width:100%;display:block}figcaption{font-size:12px;margin:5px 0}article{background:white;padding:18px;border-radius:8px;margin:22px 0}.trace{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;max-width:850px}.overlay{position:relative}.overlay canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.pick{cursor:crosshair}table{border-collapse:collapse;background:white}td,th{padding:8px 13px;text-align:left;border-bottom:1px solid #ddd}.scroll{overflow-x:auto}summary{cursor:pointer;margin:12px 0;font-weight:600}button{font:inherit;padding:8px}small{color:#526977}@media(max-width:800px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.trace{grid-template-columns:1fr 1fr}main{padding:12px}}'''


def build(root):
    dest=root/'site';dest.mkdir(exist_ok=True);assets=dest/'assets';assets.mkdir(exist_ok=True)
    run=root/'run';resultpath=run/'RESULT.json'
    state=json.loads((root/'STATE.json').read_text()) if (root/'STATE.json').exists() else {'stage':'preparing'}
    result=json.loads(resultpath.read_text()) if resultpath.exists() else None
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LeFFA native attention · detail supervision</title><style>'+STYLE+'</style><main>',
           '<a href="../leffa-single-step/">Previous experiment</a><h1>LeFFA: actual garment attention + DINO</h1>']
    status=result['status'] if result else state['stage']
    parts.append('<div class="status"><strong>'+html.escape(status.replace('_',' '))+'</strong>')
    if not result:
        parts.append('<p>No quality result is available yet. Training and evaluation must finish before these comparisons can establish improvement.</p>')
        if state.get('error'):parts.append('<p>'+html.escape(state['error'])+'</p>')
        if state.get('gpu'):parts.append('<p>Execution device: '+html.escape(state['gpu'])+'</p>')
        if 'free_gpu_mib' in state:parts.append(f'<p>Selected GPU free memory at last check: {state["free_gpu_mib"]:,} MiB; job requires {state["required_free_mib"]:,} MiB before loading.</p>')
        if (run/'PROGRESS.json').exists():
            progress=json.loads((run/'PROGRESS.json').read_text())
            if state['stage']=='train' and progress.get('arm'):
                parts.append(f'<p>{html.escape(progress["arm"])}: {progress["step"]:,} / {progress["steps"]:,} updates; {progress["seconds_per_update"]:.2f} seconds/update.</p>')
            elif state['stage']=='cache' and progress.get('stage')=='caching':
                parts.append(f'<p>Clean feature preparation: {progress["newly_cached"]:,} newly cached / {progress["total"]:,} examples.</p>')
    parts.append('</div><p>Locations come from the same person–garment Q/K attention used by LeFFA to generate the image. Frozen DINO supervises automatic correspondence and dense image details in one sampled-timestep training forward. Named-part heads and Molmo are absent.</p>')
    if 'updated' in state:
        parts.append('<p><small>State checked '+datetime.fromtimestamp(state['updated'],timezone.utc).strftime('%Y-%m-%d %H:%M UTC')+'. The page updates when the job changes phase.</small></p>')
    if result:
        shutil.copy2(resultpath,dest/'RESULT.json')
        parts.append('<p><a href="RESULT.json">Full criteria and paired-image bootstrap statistics</a></p><div class="scroll"><table><tr><th>DINO at every timestep compared with</th><th>DINO delta ↓</th><th>Garment LPIPS delta ↓</th><th>SSIM delta ↑</th></tr>')
        for control,metrics in result['paired_image_bootstrap_deltas_dino_all_minus_control'].items():
            cells=[]
            for m in ['dino','garment_lpips','ssim']:
                s=metrics[m];cells.append('n/a' if s is None else f'{s["mean"]:+.4f} [{s["ci95"][0]:+.4f}, {s["ci95"][1]:+.4f}]')
            parts.append('<tr><td>'+html.escape(control)+'</td>'+''.join('<td>'+s+'</td>' for s in cells)+'</tr>')
        parts.append('</table></div><p>Intervals are image-bootstrap 95% intervals. Full samples use 20 DDPM steps, matched seeds, guidance 2.5, and no final RGB repaint. These sampling runs are evaluation only.</p>')
    selected=run/'DISPLAY_CASES.json'
    if selected.exists():
        selection=json.loads(selected.read_text());parts.append('<p>'+html.escape(selection['selection'])+'</p>')
        for key in selection['keys']:
            folder=assets/key;folder.mkdir(exist_ok=True)
            sources=[root/'data'/key/'garment.png',root/'data'/key/'target.png']+[run/'samples'/arm/(key+'.png') for arm in ['diffusion','correspondence','dino_all','dino_low']]
            labels=['Input garment','Real target','Diffusion control','Correspondence only','DINO · every timestep','DINO · lower-noise half']
            parts.append('<article><h2>'+html.escape(key)+'</h2><div class="grid">')
            for n,(src,label) in enumerate(zip(sources,labels)):
                shutil.copy2(src,folder/f'{n}.png')
                parts.append(f'<figure><img loading="lazy" src="assets/{key}/{n}.png" alt="{label}"><figcaption>{label}</figcaption></figure>')
            parts.append('</div>')
            traces={}
            for arm in ['diffusion','dino_all']:
                raw=np.load(run/'traces'/arm/(key+'.npz'));att=raw['attention'].astype(np.float32)
                maximum=np.maximum(att.max(0),1e-12)
                display=np.round(att/maximum[None]*255).astype(np.uint8)
                traces[arm]={'map':base64.b64encode(display.tobytes()).decode(),'source_mass':att.sum(0).tolist()}
                shutil.copy2(run/'traces'/arm/(key+'.png'),folder/(arm+'_estimate.png'))
            (folder/'trace.json').write_text(json.dumps(traces,separators=(',',':')))
            # Choose the largest baseline full-sample error, then show identical
            # target-coordinate crops for all arms. Selection cannot move per arm.
            error=np.load(run/'samples/diffusion'/(key+'_errors.npz'))['dino']
            pm=np.asarray(Image.open(root/'data'/key/'person_mask.png').resize((24,32),Image.Resampling.NEAREST))>0
            index=int(np.where(pm.flatten(),error,-1).argmax());y,x=divmod(index,24)
            cx,cy=(x+.5)*16,(y+.5)*16
            left=max(0,min(384-80,int(cx-40)));top=max(0,min(512-80,int(cy-40)))
            parts.append('<details><summary>Same detail crop in the target and all generated outputs</summary><div class="grid">')
            for n in range(1,6):
                Image.open(folder/f'{n}.png').crop((left,top,left+80,top+80)).resize((320,320)).save(folder/f'crop{n}.png')
                parts.append(f'<figure><img loading="lazy" src="assets/{key}/crop{n}.png" alt="{labels[n]} detail"><figcaption>{labels[n]}</figcaption></figure>')
            parts.append('</div><small>Selected at the highest baseline DINO-error garment cell; identical 80×80 source coordinates for every crop.</small></details>')
            parts.append(f'<details class="probe" data-key="{key}"><summary>Click any source garment patch to inspect actual generation attention</summary><p>Single training forward at t=499/999, with the same noisy target for both models. These are clean-image estimates, separate from the fully sampled images above. Orange shows native reference attention; it is not a segmentation mask.</p><div class="trace"><figure><div class="overlay pick"><img loading="lazy" src="assets/{key}/0.png" alt="Click garment patch"><canvas width="384" height="512"></canvas></div><figcaption>Input garment · click a patch</figcaption></figure>')
            for arm,label in [('diffusion','Diffusion control'),('dino_all','DINO at every timestep')]:
                parts.append(f'<figure><div class="overlay"><img loading="lazy" src="assets/{key}/{arm}_estimate.png" alt="{label} single-step estimate"><canvas class="heat" data-arm="{arm}" width="24" height="32"></canvas></div><figcaption>{label}</figcaption></figure>')
            parts.append('</div><small class="patch-status">Select a garment patch.</small></details></article>')
    parts.append('''<script>document.querySelectorAll('.probe').forEach(el=>{let data;const pick=el.querySelector('.pick');pick.addEventListener('click',async e=>{if(!data){const raw=await(await fetch('assets/'+el.dataset.key+'/trace.json')).json();data={};for(const [k,v] of Object.entries(raw)){data[k]={map:Uint8Array.from(atob(v.map),c=>c.charCodeAt(0)),mass:v.source_mass}}}const r=pick.getBoundingClientRect();const x=Math.max(0,Math.min(23,Math.floor((e.clientX-r.left)/r.width*24))),y=Math.max(0,Math.min(31,Math.floor((e.clientY-r.top)/r.height*32))),j=y*24+x;const c=pick.querySelector('canvas').getContext('2d');c.clearRect(0,0,384,512);c.strokeStyle='#ed342c';c.lineWidth=3;c.strokeRect(x*16,y*16,16,16);el.querySelectorAll('.heat').forEach(canvas=>{const ctx=canvas.getContext('2d'),im=ctx.createImageData(24,32),v=data[canvas.dataset.arm];for(let i=0;i<768;i++){im.data[4*i]=255;im.data[4*i+1]=100;im.data[4*i+2]=0;im.data[4*i+3]=Math.round(v.map[i*768+j]*.7)}ctx.putImageData(im,0,0)});el.querySelector('.patch-status').textContent='Source patch ('+x+', '+y+'). Absolute reference mass: control '+data.diffusion.mass[j].toFixed(4)+', trained '+data.dino_all.mass[j].toFixed(4)+'. Each overlay scales its own peak for visibility.'})})</script></main></html>''')
    (dest/'index.html').write_text(''.join(parts));print('REPORT',dest/'index.html')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);build(p.parse_args().root)
