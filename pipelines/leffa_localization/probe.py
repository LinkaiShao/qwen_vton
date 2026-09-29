"""Frozen LeFFA single-pass readout and offline interventions. Never optimizes weights."""
import sys
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from common import *
from attention import DetailTracker
from annotate import instance_masks
from metrics import footprint,score_map

sys.path.insert(0,str(WORKSPACE/'garment_structure_eval/leffa_native'))
from runner import load_model,require_gpu,image,save_image

def assets():
    return SimpleNamespace(leffa_code=WORKSPACE/'garment_structure_eval/benchmark/vendor/leffa',
          leffa_weights=Path('/mnt/nvme0/hf_home/hub/models--franciszzj--Leffa/snapshots/61d3390f444506f052feedb0b243cd5369c29c89'))

class Audit:
    def __init__(self,root):
        bind_gpu();require_gpu();self.root=root
        self.model,_=load_model(assets());self.model.requires_grad_(False).eval()
        assert not any(p.requires_grad for p in self.model.parameters())
        self.tracker=DetailTracker(self.model.unet)
        self.alpha=self.model.noise_scheduler.alphas_cumprod.cuda().float();self.calls={'generator':0,'reference':0}
        for label,mod in [('generator',self.model.unet),('reference',self.model.unet_encoder)]:
            def count(*_,label=label):self.calls[label]+=1
            mod.register_forward_hook(count)

    @torch.inference_mode()
    def row(self,info,old_mask=False):
        f=self.root/'data'/info['key'];meta=read(f/'IDENTITY.json')
        if meta['status']!='ready':raise ValueError('Identity mask failed')
        cache=self.root/'latent_cache'/(info['key']+('_old' if old_mask else '')+'.pt')
        maskpath=f/('old_inpaint.png' if old_mask else 'inpaint.png')
        identity='|'.join([digest(maskpath),*[digest(f/p) for p in ('target.png','garment.png','densepose.png')],
                          str(assets().leffa_weights.resolve()),'virtual_tryon.pth:deterministic_vae_mode'])
        if cache.exists():
            obj=torch.load(cache,map_location='cpu',weights_only=True)
            if obj['cache_identity']!=identity:raise ValueError('Stale latent cache')
        else:
            target=image(f/'target.png').cuda();source=image(f/'garment.png').cuda();mask=image(maskpath,True).cuda()
            pose=image(f/'densepose.png').cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                def enc(x):return self.model.vae.encode(x.to(torch.bfloat16)).latent_dist.mode()*self.model.vae.config.scaling_factor
                z0=enc(target);zg=enc(source);za=enc(target*(mask<.5))
            obj={'z0':z0.cpu(),'source':zg.cpu(),'agnostic':za.cpu(),
                 'mask':F.interpolate(mask,z0.shape[-2:],mode='nearest').cpu(),
                 'pose':F.interpolate(pose,z0.shape[-2:],mode='nearest').cpu(),'cache_identity':identity}
            cache.parent.mkdir(parents=True,exist_ok=True);torch.save(obj,cache)
        out={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in obj.items()}
        with np.load(f/'parts.npz') as parts:
            instances,names=instance_masks(parts['source'])
            out['source_parts']=torch.from_numpy(np.concatenate([parts['source'],instances]).astype(np.float32))[None].cuda()
            out['source_instances']=names
        out['meta']=meta;out['key']=info['key']
        return out

    @torch.inference_mode()
    def trace_details(self,row,t,decode=True):
        self.tracker.clear();self.tracker.parts=row['source_parts'];before=dict(self.calls)
        seed=int(sha_order(row['key']+':'+str(t))[:8],16)
        eps=torch.randn(row['z0'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed),dtype=torch.float32)
        a=self.alpha[t];noisy=a.sqrt()*row['z0'].float()+(1-a).sqrt()*eps;tt=torch.tensor([t],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,ref=self.model.unet_encoder(row['source'],tt,encoder_hidden_states=None,return_dict=False)
            prediction=self.model.unet(torch.cat([noisy,row['mask'],row['agnostic'],row['pose']],1),tt,
                                       encoder_hidden_states=None,reference_features=list(ref),return_dict=False)[0]
            rgb=self.model.vae.decode(((noisy-(1-a).sqrt()*prediction.float())/a.sqrt()).to(torch.bfloat16)/self.model.vae.config.scaling_factor).sample.float() if decode else None
        assert all(self.calls[k]-before[k]==1 for k in before),'More than one generation forward'
        maps,names=self.tracker.candidates() if self.tracker.enabled else (None,None)
        return {'prediction':prediction,'rgb':rgb,'maps':maps,'candidates':names,'seed':seed}

def verify(audit,info):
    row=audit.row(info);audit.tracker.enabled=False;x=audit.trace_details(row,499,False)
    audit.tracker.enabled=True;y=audit.trace_details(row,499,False)
    diff=float((x['prediction']-y['prediction']).abs().max())
    if diff!=0:raise AssertionError('Readout changed output')
    atomic(audit.root/'VERIFIED.json',{'passed':True,'max_output_difference':diff,'frozen_parameters':True,
           'one_generator_forward':True,'one_reference_forward':True,'candidates':y['candidates'],
           'grid':list(GRID),'finest_pixel_cell_size':8,'native_layer_grids':audit.tracker.native_grids,
           'named_region_source':'flatlay only','DINO_loaded':False,
           'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})

def smoke(root,limit=2):
    """Real frozen-model proof before the full audit: parity, five noise levels, V causality."""
    audit=Audit(root);rows=[r for r in manifest(root) if (root/'data'/r['key']/'IDENTITY.json').exists()
                          and read(root/'data'/r['key']/'IDENTITY.json')['status']=='ready'][:limit]
    if not rows:raise RuntimeError('No prepared identity masks for GPU verification')
    verify(audit,rows[0]);records=[]
    for info in rows:
        row=audit.row(info);dest=root/'smoke'/info['key'];dest.mkdir(parents=True,exist_ok=True)
        for t in TIMESTEPS:
            before=time.perf_counter();audit.tracker.enabled=True
            base=audit.trace_details(row,t);torch.cuda.synchronize();seconds=time.perf_counter()-before
            assert torch.isfinite(base['maps']).all() and torch.isfinite(base['rgb']).all()
            save_image(base['rgb'],dest/f'{t}.png')
            np.savez_compressed(dest/f'{t}.npz',maps=base['maps'].cpu().numpy(),candidates=np.array(base['candidates']))
            j=next((j for j,p in enumerate(PARTS) if row['meta']['source_parts'][p]['status']=='present'),None)
            effect=None
            if j is not None:
                audit.tracker.enabled=False;audit.tracker.set_intervention('all/mean',row['source_parts'][:,j:j+1],0.)
                changed=audit.trace_details(row,t);effect=float((changed['rgb']-base['rgb']).abs().mean())
                save_image(changed['rgb'],dest/f'{t}_intervened.png');audit.tracker.intervention=None
            records.append({'key':info['key'],'t':t,'seconds':seconds,'part_intervened':PARTS[j] if j is not None else None,
                            'output_change_mean':effect,'finite_maps':True})
            audit.tracker.enabled=True;audit.tracker.clear()
            print('SMOKE',json.dumps(records[-1]),flush=True)
    atomic(root/'SMOKE.json',{'passed':True,'records':records,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
                             'localization_accuracy_proven':False,'note':'Causal output change alone does not prove correct localization.'})

def probe(root,role,limit=None):
    audit=Audit(root);rows=[r for r in manifest(root) if r['role']==role]
    if limit:rows=rows[:limit]
    selection=read(root/'SELECTION.json') if role!='calibration' else None
    ready=[r for r in rows if read(root/'data'/r['key']/'IDENTITY.json')['status']=='ready']
    if ready and not (root/'VERIFIED.json').exists():verify(audit,ready[0])
    for n,info in enumerate(rows):
        f=root/'data'/info['key'];dest=root/'probe'/info['key'];dest.mkdir(parents=True,exist_ok=True)
        meta=read(f/'IDENTITY.json')
        if meta['status']!='ready':atomic(dest/'FAILED.json',{'reason':'identity_mask_failed'});continue
        row=audit.row(info);target=np.load(f/'parts.npz')['target']
        for t in TIMESTEPS:
            done=dest/f'{t}.json'
            if done.exists():continue
            out=audit.trace_details(row,t);maps=out['maps'].cpu().numpy();names=out['candidates']
            save_image(out['rgb'],dest/f'{t}.png')
            if selection:
                separate=[];instance_records=[]
                for i,name in enumerate(row['source_instances']):
                    p=name.split('_source_instance_')[0];m=maps[names.index(selection[p]['candidate']),len(PARTS)+i]
                    separate.append(m);fp=footprint(m);fp.pop('region')
                    instance_records.append({'name':name,'footprint':fp,'confidence_calibrated':False})
                maps=np.stack([maps[names.index(selection[p]['candidate']),j] for j,p in enumerate(PARTS)])
                np.savez_compressed(dest/f'{t}.npz',maps=maps.astype(np.float32),
                                    instance_maps=np.array(separate,dtype=np.float32),instance_names=np.array(row['source_instances']))
            else:np.savez_compressed(dest/f'{t}.npz',maps=maps.astype(np.float32),candidates=np.array(names))
            scores={}
            for j,p in enumerate(PARTS):
                eligible=meta['source_parts'][p]['status']=='present' and meta['target_parts'][p]['status']=='present'
                if selection:
                    fp=footprint(maps[j]);fp.pop('region')
                    scores[p]={'source_status':meta['source_parts'][p]['status'],'target_status':meta['target_parts'][p]['status'],
                               'footprint':fp,'confident':meta['source_parts'][p]['status']=='present' and fp['confidence']>=selection[p]['threshold'],
                               'intended':score_map(maps[j],target[j]) if eligible else None}
                else:scores[p]={name:score_map(maps[i,j],target[j]) if eligible else None for i,name in enumerate(names)}
            atomic(done,{'key':info['key'],'role':role,'t':t,'bin':t//200,'seed':out['seed'],'scores':scores,'single_forward':True,
                         'individual_sleeve_ends':instance_records if selection else [],
                         'individual_instance_confidence':'Uncalibrated diagnostic readouts; never merge these into a DINO crop spanning the torso.'})
            audit.tracker.clear()
        atomic(root/'PROBE_PROGRESS.json',{'role':role,'done':n+1,'total':len(rows),'key':info['key']})
        print('PROBED',role,n+1,len(rows),info['key'],flush=True)

def calibrate(root):
    selected={}
    for p in PARTS:
        records=[]
        for r in manifest(root):
            if r['role']!='calibration':continue
            for t in TIMESTEPS:
                path=root/'probe'/r['key']/f'{t}.json'
                if path.exists():records.append(read(path))
        names=read(root/'VERIFIED.json')['candidates'];ranking=[]
        for name in names:
            bybin=[[r['scores'][p][name] for r in records if r['bin']==b and r['scores'][p].get(name) is not None] for b in range(5)]
            qualities=[np.mean([m['inside_mass']*float(m['peak_hit']) for m in group]) if group else 0 for group in bybin]
            ranking.append((min(qualities),float(np.mean(qualities)),name))
        _,_,name=max(ranking)
        eligible=[r['scores'][p][name] for r in records if r['scores'][p].get(name) is not None]
        threshold=1.1;coverage=0.
        for value in sorted({0.,*[m['confidence'] for m in eligible]}):
            accepted=[m for m in eligible if m['confidence']>=value]
            if accepted and np.mean([m['peak_hit'] for m in accepted])>=.9 and np.mean([m['inside_mass'] for m in accepted])>=.8:
                threshold=value;coverage=len(accepted)/max(1,len(eligible));break
        selected[p]={'candidate':name,'threshold':threshold,'calibration_coverage':coverage,
                     'calibration_observations':len(eligible),'calibration_passed':coverage>=.8,
                     'unfiltered_peak_hit':float(np.mean([m['peak_hit'] for m in eligible])) if eligible else 0.,
                     'unfiltered_inside_mass':float(np.mean([m['inside_mass'] for m in eligible])) if eligible else 0.,
                     'rule':'Maximize worst-noise-bin mean inside-mass times peak-hit, then overall mean; fixed for all heldout images/timesteps.'}
    atomic(root/'SELECTION.json',selected);print('CALIBRATED',json.dumps(selected),flush=True)

def selected_map(out,selection,j):return out['maps'][out['candidates'].index(selection[PARTS[j]]['candidate']),j]

def unrelated_region(part,garment):
    """Deterministic compact, equal-area source control, separate from the part."""
    from scipy.ndimage import distance_transform_edt
    mask=part[0,0].cpu().numpy()>.5;fg=garment[0,0].cpu().numpy()>.5;outside=fg&~mask
    count=int(mask.sum());control=np.zeros_like(mask)
    if count==0 or outside.sum()<count:return torch.zeros_like(part),False
    distance=distance_transform_edt(~mask);distance[~outside]=-1
    cy,cx=np.unravel_index(distance.argmax(),distance.shape);y,x=np.where(outside)
    take=np.argsort((y-cy)**2+(x-cx)**2,kind='stable')[:count];control[y[take],x[take]]=True
    return torch.from_numpy(control).to(part.device,dtype=part.dtype)[None,None],True

def sweep(root):
    audit=Audit(root);selection=read(root/'SELECTION.json')
    rows=[r for r in manifest(root) if r['role']=='calibration' and read(root/'data'/r['key']/'IDENTITY.json')['status']=='ready'][:16]
    if not rows:raise RuntimeError('No corrected-mask development cases')
    records=[];start=time.time()
    with (root/'timestep_sweep.jsonl').open('w') as log:
        for t in range(1000):
            info=rows[t%len(rows)];row=audit.row(info);out=audit.trace_details(row,t,False)
            target=np.load(root/'data'/info['key']/'parts.npz')['target'];scores={}
            for j,p in enumerate(PARTS):
                m=selected_map(out,selection,j).cpu().numpy()
                if not np.isfinite(m).all():raise FloatingPointError('Nonfinite mark')
                if row['meta']['source_parts'][p]['status']=='present' and row['meta']['target_parts'][p]['status']=='present':scores[p]=score_map(m,target[j])
            record={'t':t,'key':info['key'],'scores':scores};records.append(record)
            log.write(json.dumps(record)+'\n');log.flush();audit.tracker.clear()
            if t%50==0:print('SWEEP',t,flush=True)
    atomic(root/'SWEEP.json',{'timesteps':1000,'finite':True,'cases':len(rows),'seconds':time.time()-start,
                            'protocol':'Each actual timestep tested once, rotating the same 16 development cases; finite maps alone do not prove localization.'})

def causal(root):
    audit=Audit(root);selection=read(root/'SELECTION.json')
    for n,info in enumerate(manifest(root)):
        if info['role'] not in ('heldout','regression'):continue
        f=root/'data'/info['key'];meta=read(f/'IDENTITY.json')
        if meta['status']!='ready':continue
        row=audit.row(info);dest=root/'causal'/info['key'];dest.mkdir(parents=True,exist_ok=True)
        gm=image(f/'garment_mask.png',True).cuda()
        for t in TIMESTEPS:
            done=dest/f'{t}.json'
            if done.exists():continue
            base=audit.trace_details(row,t);audit.tracker.enabled=False
            repeat=audit.trace_details(row,t);repeat_energy=float((base['rgb']-repeat['rgb']).abs().mean())
            records={}
            for j,p in enumerate(PARTS):
                if meta['source_parts'][p]['status']!='present':continue
                selected=selected_map(base,selection,j).cpu().numpy();fp=footprint(selected);support=fp['region']
                candidate=selection[p]['candidate'];part=row['source_parts'][:,j:j+1]
                if not part.any():continue
                unrelated,has_control=unrelated_region(part,gm)
                interventions={}
                for label,mask,factor in [('remove',part,0.),('increase',part,1.25),('unrelated',unrelated,0.)]:
                    if label=='unrelated' and not has_control:continue
                    audit.tracker.set_intervention(candidate,mask,factor)
                    changed=audit.trace_details(row,t)
                    diff=(changed['rgb']-base['rgb']).abs().mean(1,keepdim=True)
                    effect=F.interpolate(diff,GRID,mode='area')[0,0].cpu().numpy()
                    total=float(effect.sum());hit=float(effect[support].sum()/max(total,1e-15))
                    controls=[np.roll(support,shift,axis=(0,1)) for shift in [(21,13),(-21,13),(21,-13),(-21,-13)]]
                    control=float(np.mean([effect[c].sum()/max(total,1e-15) for c in controls]))
                    record={'localized_fraction':hit,'shifted_fraction':control,'delta':hit-control,
                            'effect_mean':float(diff.mean()),'signal_above_repeat':float(diff.mean())>max(1e-6,repeat_energy*10),
                            'support_fraction':float(support.mean())}
                    interventions[label]=record
                    if label=='remove':
                        np.savez_compressed(dest/f'{t}_{p}.npz',effect=effect,footprint=selected,support=support)
                        save_image(changed['rgb'],dest/f'{t}_{p}.png')
                    audit.tracker.intervention=None
                records[p]={'candidate':candidate,'repeat_mean':repeat_energy,'interventions':interventions,
                            'equal_area_source_control':has_control,'source_control_pixels':int(unrelated.sum()),
                            'detail_pixels':int(part.sum())}
            audit.tracker.enabled=True;audit.tracker.intervention=None;audit.tracker.clear()
            atomic(done,{'key':info['key'],'t':t,'bin':t//200,'parts':records,'repeat_mean':repeat_energy})
        print('CAUSAL',n+1,info['key'],flush=True)

def compare_masks(root):
    audit=Audit(root)
    for info in manifest(root):
        if info['role']!='regression' or read(root/'data'/info['key']/'IDENTITY.json')['status']!='ready':continue
        for old in (True,False):
            row=audit.row(info,old);audit.tracker.enabled=False
            out=audit.trace_details(row,499);save_image(out['rgb'],root/'mask_comparison'/info['key']/('old.png' if old else 'corrected.png'))
    print('MASK_COMPARISON_COMPLETE',flush=True)

if __name__=='__main__':
    p=arguments(__doc__);p.add_argument('stage',choices=['probe','calibrate','sweep','causal','compare-masks','smoke'])
    p.add_argument('--role',choices=['calibration','heldout','regression'],default='calibration');p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.stage=='probe':probe(a.root,a.role,a.limit)
    elif a.stage=='calibrate':calibrate(a.root)
    elif a.stage=='sweep':sweep(a.root)
    elif a.stage=='causal':causal(a.root)
    elif a.stage=='smoke':smoke(a.root,a.limit or 2)
    else:compare_masks(a.root)
