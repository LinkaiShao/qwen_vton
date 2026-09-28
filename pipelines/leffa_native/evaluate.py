"""Independent causal interventions and full sampling. Never part of training loss."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from runner import Experiment, ARMS, SEED, image, save_image, atomic, require_h200
from core import GRID


def bootstrap(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return None
    rng=np.random.default_rng(SEED)
    means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(2000)])
    return {'mean':float(values.mean()),'ci95':[float(x) for x in np.quantile(means,[.025,.975])],'n':len(values)}


def ssim(pred,target):
    x=((pred.float()+1)/2).clamp(0,1);y=((target.float()+1)/2).clamp(0,1)
    c=torch.arange(11,device=x.device)-5
    kernel=torch.exp(-c.square()/4.5);kernel=kernel/kernel.sum()
    w=(kernel[:,None]*kernel[None,:])[None,None].repeat(3,1,1,1)
    def conv(z):return F.conv2d(z,w,groups=3)
    mx,my=conv(x),conv(y)
    vx=conv(x*x)-mx*mx;vy=conv(y*y)-my*my;vxy=conv(x*y)-mx*my
    return (((2*mx*my+.01**2)*(2*vxy+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean()


@torch.no_grad()
def causal(exp,arm,count=32):
    dest=exp.args.output/'causal'/arm;dest.mkdir(parents=True,exist_ok=True)
    records=[]
    for info in exp.rows['development'][:count]:
        row=exp.row(info)
        for t in [99,299,499,699,899]:
            stem=info['key']+'_'+str(t);resultpath=dest/(stem+'.json')
            if resultpath.exists():records.append(json.loads(resultpath.read_text()));continue
            exp.tracker.intervention=None
            _,_,x0,a=exp.predict(row,t,SEED+t)
            pm=row['person_mask'][0];gm=row['garment_mask'][0]
            eligible=torch.where(gm)[0]
            if not len(eligible) or not pm.any():continue
            # Source choice is fixed from the manifest key, not outcome quality.
            index=int(hashlib.sha256(stem.encode()).hexdigest()[:8],16)%len(eligible)
            source=int(eligible[index]);footprint=a[0,:,source].clone()
            _,error,_,rgb=exp.detail(row,x0,a)
            # Top-area support is fixed to 10% of visible target cells.
            n=max(1,int(pm.sum()*.1));support=torch.zeros_like(pm)
            support[(footprint*pm).topk(n).indices]=True
            shuffled=footprint.reshape(32,24).roll((11,7),(0,1)).flatten()*pm
            random_support=torch.zeros_like(pm);random_support[shuffled.topk(n).indices]=True
            exp.tracker.intervention=('*',source)
            _,_,changed,_=exp.predict(row,t,SEED+t)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                changed_rgb=exp.model.vae.decode(changed.to(torch.bfloat16)/exp.model.vae.config.scaling_factor).sample.float()
            exp.tracker.intervention=None
            effect=F.interpolate((changed_rgb-rgb).abs().mean(1,keepdim=True),(32,24),mode='area').flatten()
            energy=float(effect.sum())
            hit=float(effect[support].sum()/effect.sum().clamp_min(1e-12))
            control=float(effect[random_support].sum()/effect.sum().clamp_min(1e-12))
            result={'key':info['key'],'t':t,'bin':t//200,'source_index':source,'effect_energy':energy,
                    'localized_fraction':hit,'shuffled_fraction':control,'delta':hit-control,
                    'source_reference_mass':float(footprint.sum()),'support_cells':n}
            atomic(resultpath,result);records.append(result)
            if t==499:
                np.savez_compressed(dest/(info['key']+'.npz'),footprint=footprint.cpu().numpy(),
                                    effect=effect.cpu().numpy(),dino_error=error[0].cpu().numpy(),source=source)
                save_image(rgb,dest/(info['key']+'_estimate.png'))
        print('CAUSAL',arm,info['key'],flush=True)
    stats={str(b):bootstrap([r['delta'] for r in records if r['bin']==b]) for b in range(5)}
    result={'records':records,'by_noise_bin':stats,'passed':all(s is not None and s['ci95'][0]>0 for s in stats.values()),
            'interpretation':'Controlled native-value intervention, not a proof that every contextual source feature contains only local RGB.'}
    atomic(dest/'RESULT.json',result);return result


@torch.no_grad()
def full_samples(exp,arm,lpips_model):
    from leffa.pipeline import LeffaPipeline
    pipeline=LeffaPipeline(exp.model)
    exp.tracker.enabled=False
    dest=exp.args.output/'samples'/arm;dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for info in exp.rows['test']:
        metrics=dest/(info['key']+'.json')
        if metrics.exists():rows.append(json.loads(metrics.read_text()));continue
        row=exp.row(info);folder=exp.args.data/'data'/info['key']
        target=image(folder/'target.png').cuda();garment=image(folder/'garment.png').cuda()
        mask=image(folder/'mask.png',True).cuda();pose=image(folder/'densepose.png').cuda()
        seed=int(hashlib.sha256(('native-test:'+info['key']).encode()).hexdigest()[:8],16)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=pipeline(src_image=target,ref_image=garment,mask=mask,densepose=pose,
                         num_inference_steps=20,do_classifier_free_guidance=True,guidance_scale=2.5,
                         generator=torch.Generator(device='cuda').manual_seed(seed),repaint=False)[0][0]
        out.save(dest/(info['key']+'.png'))
        pred=image(dest/(info['key']+'.png')).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):features=exp.dino(pred)
        error=1-(features*F.normalize(row['target_features'].float(),dim=-1)).sum(-1)
        pm=row['person_mask'];has=bool(pm.any())
        lpips_map=lpips_model(pred.float(),target.float())
        local=F.interpolate(lpips_map,GRID,mode='bilinear',align_corners=False).flatten(1)
        record={'key':info['key'],'seed':seed,'has_foreground':has,
                'dino':float((error*pm).sum()/pm.sum()) if has else None,
                'lpips':float(lpips_map.mean()),'garment_lpips':float((local*pm).sum()/pm.sum()) if has else None,
                'ssim':float(ssim(pred,target))}
        atomic(metrics,record);np.savez_compressed(dest/(info['key']+'_errors.npz'),dino=error[0].cpu().numpy(),lpips=local[0].cpu().numpy())
        rows.append(record);print('FULL_SAMPLE',arm,info['key'],flush=True)
    exp.tracker.enabled=True
    atomic(dest/'RESULT.json',{'arm':arm,'records':rows,'sampling_steps':20,'guidance':2.5,'repaint':False})
    return rows


def compare(output):
    results={a:json.loads((output/'samples'/a/'RESULT.json').read_text())['records'] for a in ['initial',*ARMS]}
    comparisons={}
    for control in ['initial','diffusion','correspondence','dino_low']:
        c={r['key']:r for r in results[control]};d={r['key']:r for r in results['dino_all']}
        comparisons[control]={metric:bootstrap([d[k][metric]-c[k][metric] for k in c
                                if d[k][metric] is not None and c[k][metric] is not None])
                              for metric in ['dino','garment_lpips','lpips','ssim']}
    causal_result=json.loads((output/'causal/dino_all/RESULT.json').read_text())
    quality=all(comparisons[c][m] is not None and comparisons[c][m]['ci95'][1]<0
                for c in ['diffusion','correspondence'] for m in ['dino','garment_lpips'])
    all_t_advantage=all(comparisons['dino_low'][m] is not None and comparisons['dino_low'][m]['ci95'][1]<0
                       for m in ['dino','garment_lpips'])
    sweep=json.loads((output/'TIMESTEP_VERIFICATION.json').read_text())
    criteria={'causal_localization_all_noise_bands':causal_result['passed'],
              'heldout_dino_and_independent_lpips_improvement':quality,'all_timestep_gradients_finite':sweep['passed'],
              'all_timestep_advantage_over_low_noise_only':all_t_advantage,
              'native_gradient_path_verified':json.loads((output/'GRADIENT_VERIFICATION.json').read_text())['passed'],
              'training_complete':all((output/a/'COMPLETE.json').exists() for a in ARMS)}
    atomic(output/'RESULT.json',{'status':'SUCCESSFUL' if all(criteria.values()) else 'COMPLETED_CRITERIA_NOT_ALL_MET',
                               'criteria':criteria,'paired_image_bootstrap_deltas_dino_all_minus_control':comparisons,
                               'test_images':len(results['dino_all'])})


@torch.no_grad()
def traces(exp):
    """Real single-forward traces on final-test cases selected transparently."""
    initial=json.loads((exp.args.output/'samples/diffusion/RESULT.json').read_text())['records']
    joint={r['key']:r for r in json.loads((exp.args.output/'samples/dino_all/RESULT.json').read_text())['records']}
    pairs=sorted([(joint[r['key']]['dino']-r['dino'],r['key']) for r in initial if r['dino'] is not None and joint[r['key']]['dino'] is not None])
    keys=list(dict.fromkeys([k for _,k in pairs[:8]]+[k for _,k in pairs[-8:]]+[r['key'] for r in exp.rows['test'][:8]]))
    atomic(exp.args.output/'DISPLAY_CASES.json',{'keys':keys,'selection':'Eight biggest DINO improvements, eight biggest regressions, and first eight preregistered test cases; duplicates removed.'})
    for arm in ['diffusion','dino_all']:
        exp.restore(exp.args.output/arm/'latest.pt');exp.tracker.enabled=True
        for info in exp.rows['test']:
            if info['key'] not in keys:continue
            dest=exp.args.output/'traces'/arm;dest.mkdir(parents=True,exist_ok=True)
            row=exp.row(info);_,_,x0,a=exp.predict(row,499,SEED+499)
            _,error,_,rgb=exp.detail(row,x0,a)
            np.savez_compressed(dest/(info['key']+'.npz'),attention=a[0].cpu().numpy().astype(np.float16),error=error[0].cpu().numpy())
            save_image(rgb,dest/(info['key']+'.png'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['data','cache','output','leffa-code','leffa-weights','dino-weights']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--accumulation',type=int,default=4)
    args=p.parse_args();require_h200()
    import lpips
    metric=lpips.LPIPS(net='alex',spatial=True).cuda().eval().requires_grad_(False)
    exp=Experiment(args)
    for arm in ['initial',*ARMS]:
        exp.reset()
        if arm!='initial':exp.restore(args.output/arm/'latest.pt')
        if arm in ['initial','dino_all']:causal(exp,arm)
        full_samples(exp,arm,metric)
    compare(args.output)
    traces(exp)
