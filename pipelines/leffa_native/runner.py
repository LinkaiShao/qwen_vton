"""H200 worker: cache clean targets, verify real routing, train four matched arms."""
from __future__ import annotations
import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from core import (GRID, LAYERS, NativeAttentionTracker, DenseDINO, reliable_matches,
                  correspondence_loss, dense_detail_loss, reconstruct_x0, noise_weight, routing_diagnostics)
from prepare import atomic, digest

ARMS = ('diffusion', 'correspondence', 'dino_all', 'dino_low')
SEED = 20260928


def image(path, mask=False):
    a=np.asarray(Image.open(path).convert('L' if mask else 'RGB')).copy()
    t=torch.from_numpy(a).float()/255
    return t[None,None] if mask else (t.permute(2,0,1)[None]*2-1)


def save_image(t,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    a=((t.detach()[0].float().cpu().permute(1,2,0)+1)*127.5).clamp(0,255).byte().numpy()
    Image.fromarray(a).save(path)


def save_tensor(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


def require_h200():
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():
        raise RuntimeError('This worker is restricted to the authorized H200')
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def load_model(args):
    sys.path.insert(0,str(args.leffa_code))
    from leffa.model import LeffaModel
    from diffusers import DDPMScheduler
    with torch.device('meta'):
        model=LeffaModel(str(args.leffa_weights/'stable-diffusion-inpainting'),dtype='float32',height=512,width=384)
    # Scheduler tensors are not model parameters/state_dict entries. Recreate
    # outside the meta context, otherwise alphas_cumprod remains unmaterialized.
    model.noise_scheduler=DDPMScheduler.from_pretrained(args.leffa_weights/'stable-diffusion-inpainting',subfolder='scheduler')
    state=torch.load(args.leffa_weights/'virtual_tryon.pth',map_location='cpu',mmap=True,weights_only=True)
    model.load_state_dict(state,assign=True);del state
    model.to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    params={}
    for name,module in model.unet.named_modules():
        if name.startswith(('up_blocks.2.','up_blocks.3.')) and name.endswith('.attn1'):
            for field in ['to_q','to_k','to_v','to_out.0']:
                projection=module.get_submodule(field).float().requires_grad_(True)
                for p_name,p in projection.named_parameters():params[name+'.'+field+'.'+p_name]=p
    if not params:raise RuntimeError('No native generative parameters selected')
    return model,params


@torch.no_grad()
def cache(args):
    from diffusers import AutoencoderKL
    args.cache.mkdir(parents=True,exist_ok=True)
    config={'manifest_sha256':digest(args.data/'manifest.json'),'grid':list(GRID),
            'dino_weights':args.dino_weights.name,'leffa_weights':args.leffa_weights.name,
            'match_cosine_min':.5,'cycle_distance_max':1,'code_sha256':digest(Path(__file__).with_name('core.py'))}
    cp=args.cache/'CONFIG.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Incompatible feature cache')
    atomic(cp,config)
    with torch.device('meta'):
        cfg=AutoencoderKL.load_config(args.leffa_weights/'stable-diffusion-inpainting',subfolder='vae')
        vae=AutoencoderKL.from_config(cfg)
    weights=torch.load(args.leffa_weights/'virtual_tryon.pth',map_location='cpu',mmap=True,weights_only=True)
    vae.load_state_dict({k[4:]:v for k,v in weights.items() if k.startswith('vae.')},assign=True);del weights
    vae.to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    dino=DenseDINO(str(args.dino_weights))
    manifest=json.loads((args.data/'manifest.json').read_text());records=manifest['records']
    start=time.time();done=0;coverage=[]
    for row in records:
        dest=args.cache/(row['key']+'.pt')
        if dest.exists():continue
        folder=args.data/'data'/row['key']
        # Upload may still be working, but never consume an incomplete example.
        if not (folder/'READY.json').exists():raise RuntimeError('Incomplete data '+row['key'])
        target=image(folder/'target.png').cuda();garment=image(folder/'garment.png').cuda()
        mask=image(folder/'mask.png',True).cuda();pose=image(folder/'densepose.png').cuda()
        pm=(F.interpolate(image(folder/'person_mask.png',True).cuda(),GRID,mode='area').flatten(1)>.5)[0]
        gm=(F.interpolate(image(folder/'garment_mask.png',True).cuda(),GRID,mode='area').flatten(1)>.5)[0]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            def encode(x):return vae.encode(x.to(torch.bfloat16)).latent_dist.mode()*vae.config.scaling_factor
            z0=encode(target);zg=encode(garment);za=encode(target*(mask<.5))
            tf=dino(target);gf=dino(garment)
        teacher=reliable_matches(tf[0],gf[0],pm,gm)
        obj={'key':row['key'],'source_hashes':row['source_hashes'],
             'z0':z0.cpu(),'garment':zg.cpu(),'agnostic':za.cpu(),
             'mask':F.interpolate(mask,z0.shape[-2:],mode='nearest').cpu().half(),
             'pose':F.interpolate(pose,z0.shape[-2:],mode='nearest').cpu().half(),
             'target_features':tf.cpu().half(),'person_mask':pm[None].cpu(),
             'garment_mask':gm[None].cpu(),**{k:v[None] for k,v in teacher.items()}}
        save_tensor(dest,obj);done+=1;coverage.append(float(teacher['reliable'].sum()/pm.sum().clamp_min(1)))
        if done%25==0:
            progress={'stage':'caching','newly_cached':done,'total':len(records),'last':row['key'],
                      'seconds':time.time()-start,'mean_reliable_fraction':float(np.mean(coverage)),
                      'peak_gib':torch.cuda.max_memory_allocated()/2**30}
            atomic(args.output/'PROGRESS.json',progress);print('CACHE',json.dumps(progress),flush=True)
    missing=[r['key'] for r in records if not (args.cache/(r['key']+'.pt')).exists()]
    if missing:raise RuntimeError('Incomplete cache')
    atomic(args.cache/'READY.json',dict(config,count=len(records),seconds=time.time()-start))
    print('CACHE_COMPLETE',len(records),flush=True)


class Experiment:
    def __init__(self,args):
        self.args=args
        self.manifest=json.loads((args.data/'manifest.json').read_text())
        self.rows={s:[r for r in self.manifest['records'] if r['split']==s] for s in ['train','development','test']}
        self.model,self.params=load_model(args)
        self.initial={k:v.detach().cpu().clone() for k,v in self.params.items()}
        self.tracker=NativeAttentionTracker(self.model.unet)
        self.dino=DenseDINO(str(args.dino_weights))
        self.alpha=self.model.noise_scheduler.alphas_cumprod.cuda().float()
        if self.model.noise_scheduler.config.prediction_type!='epsilon':raise ValueError('Expected DDPM epsilon checkpoint')
        self.calls={'unet':0,'reference':0}
        for label,mod in [('unet',self.model.unet),('reference',self.model.unet_encoder)]:
            def count(*_,label=label):self.calls[label]+=1
            mod.register_forward_hook(count)

    def row(self,row):
        cached=torch.load(self.args.cache/(row['key']+'.pt'),map_location='cpu',weights_only=True)
        if cached['source_hashes']!=row['source_hashes']:raise RuntimeError('Source/cache mismatch')
        return {k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in cached.items()}

    def reset(self):
        with torch.no_grad():
            for k,p in self.params.items():p.copy_(self.initial[k].to(p.device))
        self.tracker.clear();self.tracker.intervention=None
        for p in self.params.values():p.grad=None

    def restore(self,path):
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        if set(ckpt['parameters'])!=set(self.params):raise RuntimeError('Native checkpoint parameter mismatch')
        with torch.no_grad():
            for k,p in self.params.items():p.copy_(ckpt['parameters'][k].to(p.device))
        return ckpt

    def predict(self,row,t,seed):
        self.tracker.clear();before=dict(self.calls)
        generator=torch.Generator(device='cuda').manual_seed(seed)
        eps=torch.randn(row['z0'].shape,device='cuda',generator=generator,dtype=torch.float32)
        a=self.alpha[t].reshape(1,1,1,1)
        noisy=a.sqrt()*row['z0'].float()+(1-a).sqrt()*eps
        tt=torch.tensor([t],device='cuda',dtype=torch.long)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            _,features=self.model.unet_encoder(row['garment'],tt,encoder_hidden_states=None,return_dict=False)
        inputs=torch.cat((noisy,row['mask'],row['agnostic'],row['pose']),1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=self.model.unet(inputs,tt,encoder_hidden_states=None,reference_features=list(features),return_dict=False)[0]
        if any(self.calls[k]-before[k]!=1 for k in before):raise RuntimeError('Training prediction used more than one model forward')
        return pred,eps,reconstruct_x0(noisy,pred,a),self.tracker.mean() if self.tracker.enabled else None

    def detail(self,row,x0,a):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            rgb=self.model.vae.decode(x0.to(torch.bfloat16)/self.model.vae.config.scaling_factor).sample.float()
            features=self.dino(rgb)
        loss,error,weights=dense_detail_loss(features,row['target_features'],a,row['person_mask'],row['garment_mask'])
        return loss,error,weights,rgb

    def corr(self,row,a):
        return torch.stack([correspondence_loss(v,row['match'],row['reliable'],row['garment_mask'])
                            for v in self.tracker.maps.values()]).mean()

    def norms(self,grads):
        return {kind:float(sum((g.detach().float().square().sum() for (name,_),g in zip(self.params.items(),grads)
                               if g is not None and kind in name),torch.tensor(0.,device='cuda')).sqrt())
                for kind in ['to_q','to_k','to_v','to_out']}

    def verify(self):
        row=self.row(next(r for r in self.rows['development'] if self.row(r)['reliable'].any()))
        with torch.no_grad():
            self.tracker.enabled=False;original=self.predict(row,500,10)[0]
            self.tracker.enabled=True;traced=self.predict(row,500,10)[0]
            delta=float((original-traced).abs().max())
        if delta!=0:raise AssertionError('Tracer changed denoiser output')
        pred,eps,x0,a=self.predict(row,500,10)
        corr=self.corr(row,a)
        cg=torch.autograd.grad(corr,list(self.params.values()),retain_graph=True,allow_unused=True)
        cn=self.norms(cg);del cg
        loss,_,_,_=self.detail(row,x0,a)
        dg=torch.autograd.grad(loss,list(self.params.values()),allow_unused=True)
        dn=self.norms(dg);del dg
        if not all(math.isfinite(v) and v>0 for v in dn.values()) or cn['to_q']<=0 or cn['to_k']<=0:
            raise AssertionError(f'Missing/nonfinite native gradient: {cn}, {dn}')
        frozen=all(not p.requires_grad for module in [self.model.vae,self.model.unet_encoder,self.dino] for p in module.parameters())
        if not frozen:raise AssertionError('Teacher/VAE/reference encoder not frozen')
        result={'output_max_difference':delta,'correspondence_gradient_norms':cn,'dino_gradient_norms':dn,
                'parameters':sum(p.numel() for p in self.params.values()),'parameter_names':list(self.params),
                'frozen_networks_verified':True,'model_calls_per_prediction':{'unet':1,'reference':1},
                'peak_gib':torch.cuda.max_memory_allocated()/2**30,'passed':True}
        atomic(self.args.output/'GRADIENT_VERIFICATION.json',result)
        print('GRADIENT_VERIFIED',json.dumps({k:v for k,v in result.items() if k!='parameter_names'}),flush=True)
        self.tracker.clear();gc.collect();torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(self,tag,count=64):
        records=[];start=time.time()
        for rowinfo in self.rows['development'][:count]:
            row=self.row(rowinfo)
            for t in [99,299,499,699,899]:
                pred,eps,x0,a=self.predict(row,t,SEED+t)
                dino,error,weights,rgb=self.detail(row,x0,a)
                pm=row['person_mask'];valid=row['reliable'];gm=row['garment_mask']
                masked=a*gm[:,None]
                arg=masked.argmax(-1)
                dist=torch.stack((arg//24-row['match']//24,arg%24-row['match']%24),-1).float().norm(dim=-1)
                refmass=(a.sum(-1)*pm).sum()/pm.sum().clamp_min(1)
                diagnostic=routing_diagnostics(a,gm)
                records.append({'key':rowinfo['key'],'t':t,'bin':t//200,'dino':float(dino),
                                'uniform_dino':float((error*pm).sum()/pm.sum().clamp_min(1)),
                                'has_foreground':bool(pm.any()),'pck_1cell':float(((dist<=1)*valid).sum()/valid.sum().clamp_min(1)),
                                'reliable_count':int(valid.sum()),'reference_mass':float(refmass),
                                'conditional_entropy':float((diagnostic['conditional_entropy']*pm).sum()/pm.sum().clamp_min(1)),
                                'epsilon_mse':float(F.mse_loss(pred.float(),eps))})
        path=self.args.output/'evaluations'/(tag+'.json')
        atomic(path,{'tag':tag,'records':records,'seconds':time.time()-start})
        print('EVALUATED',tag,len(records),flush=True)
        return records

    def timestep_sweep(self):
        path=self.args.output/'timestep_sweep.jsonl';start=time.time()
        # Each scheduler timestep is tested on a real example, rotating 16 cases.
        with path.open('w') as f:
            for t in range(len(self.alpha)):
                row=self.row(self.rows['development'][t%16])
                _,_,x0,a=self.predict(row,t,SEED+t)
                dino,_,_,_=self.detail(row,x0,a);corr=self.corr(row,a)
                grads=torch.autograd.grad(noise_weight(self.alpha[t])*dino+.1*corr,list(self.params.values()),allow_unused=True)
                norms=self.norms(grads)
                if not torch.isfinite(dino+corr) or not all(math.isfinite(v) for v in norms.values()) or sum(norms.values())<=0:
                    raise RuntimeError('Missing/nonfinite timestep gradient '+str(t))
                f.write(json.dumps({'t':t,'weight':float(noise_weight(self.alpha[t])),'dino':float(dino.detach()),'gradient_norms':norms})+'\n');f.flush()
                del grads,dino,corr,x0,a;self.tracker.clear()
                if t%50==0:print('TIMESTEP_SWEEP',t,flush=True)
        atomic(self.args.output/'TIMESTEP_VERIFICATION.json',{'timesteps':len(self.alpha),'passed':True,'seconds':time.time()-start})

    def train(self,arm):
        out=self.args.output/arm;out.mkdir(parents=True,exist_ok=True)
        if (out/'COMPLETE.json').exists():return
        self.reset();optimizer=torch.optim.AdamW(list(self.params.values()),lr=1e-5,weight_decay=.01)
        start_step=0
        if (out/'latest.pt').exists():
            ckpt=self.restore(out/'latest.pt');optimizer.load_state_dict(ckpt['optimizer']);start_step=ckpt['step']
        rng=random.Random(SEED);schedule=[]
        while len(schedule)<self.args.steps*self.args.accumulation:
            indices=list(range(len(self.rows['train'])));rng.shuffle(indices);schedule.extend(indices)
        schedule_hash=hashlib.sha256(json.dumps(schedule[:self.args.steps*self.args.accumulation]).encode()).hexdigest()
        start=time.time();completed=0
        with (out/'train.jsonl').open('a') as log:
            for step in range(start_step,self.args.steps):
                optimizer.zero_grad(set_to_none=True);metrics=[]
                for micro in range(self.args.accumulation):
                    index=step*self.args.accumulation+micro;seed=SEED+index*7919
                    rowinfo=self.rows['train'][schedule[index]];row=self.row(rowinfo)
                    t=random.Random(seed).randrange(len(self.alpha))
                    pred,eps,x0,a=self.predict(row,t,seed)
                    mse=F.mse_loss(pred.float(),eps);corr=self.corr(row,a)
                    use_dino=arm=='dino_all' or (arm=='dino_low' and t<500)
                    with contextlib.nullcontext() if use_dino else torch.no_grad():
                        detail,error,weights,_=self.detail(row,x0,a)
                    dw=float(noise_weight(self.alpha[t])) if use_dino else 0.
                    loss=mse+(.1*corr if arm!='diffusion' else 0)+dw*detail
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
                    (loss/self.args.accumulation).backward()
                    metrics.append({'key':rowinfo['key'],'t':t,'seed':seed,'mse':float(mse.detach()),
                                    'corr':float(corr.detach()),'dino':float(detail.detach()),'dino_weight':dw,
                                    'has_foreground':bool(row['person_mask'].any()),'reliable_count':int(row['reliable'].sum())})
                    del pred,eps,x0,a,mse,corr,detail,error,weights,loss;self.tracker.clear()
                grad=torch.nn.utils.clip_grad_norm_(list(self.params.values()),1.)
                if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();completed+=1
                record={'arm':arm,'step':step+1,'microbatches':metrics,'grad_norm':float(grad),
                        'seconds':time.time()-start,'peak_gib':torch.cuda.max_memory_allocated()/2**30}
                log.write(json.dumps(record)+'\n');log.flush()
                if (step+1)%10==0:
                    progress={'stage':'training','arm':arm,'step':step+1,'steps':self.args.steps,
                              'seconds_per_update':(time.time()-start)/completed,'peak_gib':record['peak_gib']}
                    atomic(self.args.output/'PROGRESS.json',progress);print('TRAIN',json.dumps(progress),flush=True)
                if (step+1)%250==0 or step+1==self.args.steps:
                    save_tensor(out/'latest.pt',{'arm':arm,'step':step+1,'parameters':{k:p.detach().cpu() for k,p in self.params.items()},
                                               'optimizer':optimizer.state_dict(),'schedule_sha256':schedule_hash,
                                               'manifest_sha256':digest(self.args.data/'manifest.json')})
                if (step+1)%1000==0:self.evaluate(arm+'_'+str(step+1),count=64)
        del optimizer
        changed={k:float((p.detach().cpu()-self.initial[k]).norm()) for k,p in self.params.items()}
        atomic(out/'COMPLETE.json',{'arm':arm,'steps':self.args.steps,'accumulation':self.args.accumulation,
                                  'schedule_sha256':schedule_hash,'seconds_this_session':time.time()-start,
                                  'parameter_update_norms':changed,'oom_count':0})
        self.evaluate(arm+'_final',count=len(self.rows['development']))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['cache','verify','train','sweep'])
    for name in ['data','cache','output','leffa-code','leffa-weights','dino-weights']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--arm',choices=ARMS)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    require_h200()
    try:
        if args.stage=='cache':cache(args);return
        config={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'manifest_sha256':digest(args.data/'manifest.json'),
                'code_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'seed':SEED}
        atomic(args.output/('RUN_'+args.stage+('_'+args.arm if args.arm else '')+'.json'),config)
        exp=Experiment(args)
        if args.stage=='verify':exp.verify()
        elif args.stage=='sweep':exp.timestep_sweep()
        else:
            exp.verify()
            if not (args.output/'evaluations/initial.json').exists():exp.evaluate('initial')
            for arm in ([args.arm] if args.arm else ARMS):exp.train(arm)
            atomic(args.output/'TRAINING_COMPLETE.json',{'arms':[args.arm] if args.arm else list(ARMS),'quality_success':None,
                  'note':'Training completed; causal tests, independent sample evaluation and comparisons determine success.'})
    except Exception as exc:
        atomic(args.output/('FAILED_'+args.stage+'.json'),{'error':str(exc),'traceback':traceback.format_exc(),
                'oom':isinstance(exc,torch.cuda.OutOfMemoryError),'time':time.time()})
        raise


if __name__=='__main__':main()
