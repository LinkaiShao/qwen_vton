"""Fixed-coordinate transfer: compare native column peaks, layer consensus and flow inversion.

No semantic detector, target landmarks, learned readout, target-side Molmo, or optimizer.
Each timestep is read independently. This is a diagnostic experiment, not a claim of perfect transfer.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='GPU-dbb7940e-7461-387e-c0c2-a2e33f9b78ac'
os.environ['LEFFA_EXPECTED_GPU']=os.environ['CUDA_VISIBLE_DEVICES']
os.environ['OMP_NUM_THREADS']='4'
import argparse,functools,json,math,time,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from molmo_tryon import LAYERS,WEIGHTS,WORKSPACE,source_weights,atomic,sha
sys.path.insert(0,str(WORKSPACE/'garment_structure_eval/leffa_native'))
from runner import load_model,require_gpu

ROOT=Path('/mnt/nvme0/leffa_native/fixed_mark_transfer_20260929')
AUDIT=Path('/mnt/nvme0/leffa_native/molmo_view_audit_20260929')
OLD=Path('/mnt/nvme0/leffa_native/paper_top10_pinpoints_v1/molmo_tryon_05504')
SIZE=(768,1024)

class MarkTracker:
    def __init__(self,unet,points):
        self.points=points;self.hooks=[];self.qk={};self.maps={};self.flows={};self.weights={};self.enabled=True
        for name in LAYERS:
            mod=unet.get_submodule(name)
            assert mod.norm_q is None and mod.norm_k is None
            for key in ('q','k'):
                def capture(m,i,o,name=name,key=key):
                    if self.enabled:self.qk[name,key]=o[1:2]
                self.hooks.append(getattr(mod,'to_'+key).register_forward_hook(capture))
            def finish(m,i,o,name=name):
                if not self.enabled:return
                q=self.qk.pop((name,'q'));k=self.qk.pop((name,'k'));total,dim=q.shape[1:];n=total//2;h=round(math.sqrt(n*4/3));w=n//h
                assert total==2*h*w and q.shape==k.shape
                if (h,w) not in self.weights:self.weights[h,w]=source_weights(self.points,h,w,q.device)
                weights=self.weights[h,w];heads=m.heads
                q=q[:,:n].view(1,n,heads,dim//heads).transpose(1,2).float();k=k.view(1,total,heads,dim//heads).transpose(1,2).float()
                yy,xx=torch.meshgrid((torch.arange(h,device=q.device)+.5)/h,(torch.arange(w,device=q.device)+.5)/w,indexing='ij')
                coords=torch.stack((xx,yy),-1).reshape(n,2).float();moments=torch.cat((coords,coords.square().sum(-1,keepdim=True)),1)
                maps=[];flows=[]
                with torch.autocast('cuda',enabled=False):
                    for start in range(0,n,128):
                        logits=q[:,:,start:start+128]@k.transpose(-2,-1)*m.scale
                        p=logits.softmax(-1)
                        maps.append((p[...,n:]@weights.T).mean(1))
                        # Auxiliary paper-style flow: reference-conditional softmax, temperature 2.
                        # This is deliberately distinct from the actual full-key AV probabilities above.
                        conditional=(logits[...,n:]/2).softmax(-1)
                        flows.append((conditional@moments).mean(1))
                    maps=torch.cat(maps,1).transpose(1,2).reshape(1,len(self.points),h,w)
                    flows=torch.cat(flows,1).transpose(1,2).reshape(1,3,h,w)
                    self.maps[name]=F.interpolate(maps,(128,96),mode='bilinear',align_corners=False)[0]
                    self.flows[name]=F.interpolate(flows,(128,96),mode='bilinear',align_corners=False)[0]
            self.hooks.append(mod.register_forward_hook(finish))
    def read(self):
        assert set(self.maps)==set(LAYERS)
        maps=torch.stack([self.maps[x] for x in LAYERS]).float()
        p=maps/maps.sum((-2,-1),keepdim=True).clamp_min(1e-30)
        old=p.mean(0)
        # A shared rule for every point, label, image and timestep; no chosen left/right prior.
        log_consensus=p.clamp_min(1e-12).log().mean(0)
        consensus=(log_consensus-log_consensus.amax((-2,-1),keepdim=True)).exp()
        consensus/=consensus.sum((-2,-1),keepdim=True)
        flow=torch.stack([self.flows[x] for x in LAYERS]).mean(0)
        pts=torch.tensor(self.points,device=flow.device,dtype=torch.float32)
        dist=(flow[:2][None]-pts[:,:,None,None]).square().sum(1)
        methods={'baseline':old,'consensus':consensus,'flow_inverse':-dist}
        result={}
        for name,heat in methods.items():
            i=heat.flatten(1).argmax(1);xy=torch.stack(((i%96+.5)/96,(i//96+.5)/128),-1)
            peaks=p.flatten(2).argmax(-1);pxy=torch.stack(((peaks%96+.5)/96,(peaks//96+.5)/128),-1)
            agreement=((pxy-xy[None]).norm(dim=-1)<.05).float().mean(0)
            result[name]={'xy':xy.cpu().tolist(),'layer_peak_agreement_within_005':agreement.cpu().tolist()}
            if name=='flow_inverse':result[name]['source_residual']=dist.flatten(1).min(1).values.sqrt().cpu().tolist()
        out={'methods':result,'maps':torch.stack([old,consensus]).cpu().numpy(),'flow':flow.cpu().numpy()}
        self.maps.clear();self.flows.clear();return out
    def close(self):
        for h in self.hooks:h.remove()


def setup(root):
    root.mkdir(parents=True,exist_ok=True)
    original=json.loads((OLD/'source-points.json').read_text())
    points=[{'label':x['concept'],'instance':i+1,'source_xy':p} for x in original['records'] for i,p in enumerate(x['points'])]
    cases=[{'id':'train_05504_00','role':'development: known failure','source_points_hash':sha(OLD/'source-points.json'),'points':points,
       'flatlay':str(OLD/'flatlay.png'),'person':str(OLD/'person.png'),'densepose':str(OLD/'densepose.png'),'mask':str(OLD/'inpaint.png')}]
    m=json.loads((AUDIT/'manifest.json').read_text())
    for row in m['records']:
        folder=Path('/mnt/nvme0/leffa_native/localization_v2/data')/('test_'+row['id'])
        if not (folder/'IDENTITY.json').exists():continue
        meta=json.loads((folder/'IDENTITY.json').read_text())
        if meta['status']!='ready':continue
        src=AUDIT/'predictions'/f'{row["id"]}__flatlay.json';pred=json.loads(src.read_text())
        points=[{'label':x['concept'],'instance':i+1,'source_xy':p} for x in pred['records'] for i,p in enumerate(x['points'])]
        dataset=Path(row['views']['flatlay']).parents[1]
        assert sha(folder/'inpaint.png')==meta['hashes']['inpaint.png']
        cases.append({'id':'test_'+row['id'],'role':'heldout: fixed audit order, existing verified mask','source_points_hash':sha(src),'points':points,
            'flatlay':row['views']['flatlay'],'person':str(dataset/'image'/f'{row["id"]}.jpg'),
            'densepose':str(dataset/'image-densepose'/f'{row["id"]}.jpg'),'mask':str(folder/'inpaint.png')})
    for c in cases:c['image_hashes']={k:sha(c[k]) for k in ['flatlay','person','densepose','mask']}
    protocol={'created':time.time(),'cases':cases,'methods':{
        'baseline':'Equal mean of six independently spatial-normalized layer maps; per-point argmax.',
        'consensus':'Geometric mean of the same six spatial-normalized native attention maps; per-point argmax. Floor 1e-12.',
        'flow_inverse':'Six-layer mean of mean-head reference-conditional attention coordinate expectations at temperature 2; nearest inverse source coordinate.'},
        'semantic_labels_used_for_readout':False,'source_points_modified':False,'target_localizer':False,'training':False,
        'selection':'Development case plus all available ready-mask cases in previously fixed 32-case Molmo audit order. No selection by transfer quality.',
        'single_timestep_readout':True,'temporal_smoothing':False,'source_pixel_centers':'normalized XY with align_corners=False',
        'seed':42,'steps':20,'layers':list(LAYERS),'interpretation':'Experimental correspondence estimates, not certified exact correspondences. No forced separation or semantic coordinate priors.'}
    dest=root/'PROTOCOL.json'
    if dest.exists():return json.loads(dest.read_text())
    atomic(dest,protocol);return protocol

@torch.inference_mode()
def run(root):
    spec=setup(root);require_gpu();model,_=load_model(SimpleNamespace(leffa_code=WORKSPACE/'garment_structure_eval/benchmark/vendor/leffa',leffa_weights=WEIGHTS));model.requires_grad_(False).eval();model.height,model.width=1024,768
    from leffa.pipeline import LeffaPipeline
    from leffa.transform import LeffaTransform
    pipe=LeffaPipeline(model);transform=LeffaTransform(height=1024,width=768)
    start=time.time();original_step=model.noise_scheduler.step
    for ci,case in enumerate(spec['cases']):
        dest=root/case['id'];dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():continue
        for key,expected in case['image_hashes'].items():assert sha(case[key])==expected
        ims={k:Image.open(case[k]).convert('RGB').resize(SIZE,Image.Resampling.LANCZOS) for k in ['flatlay','person','densepose']}
        mask=Image.open(case['mask']).convert('L').resize(SIZE,Image.Resampling.NEAREST)
        batch=transform({'src_image':[ims['person']],'ref_image':[ims['flatlay']],'densepose':[ims['densepose']],'mask':[mask]})
        ims['flatlay'].save(dest/'flatlay.png');ims['person'].save(dest/'person.png');mask.save(dest/'inpaint.png')
        tracker=MarkTracker(model.unet,[x['source_xy'] for x in case['points']]);frames=[];maps=[];flows=[];latents={'baseline':[],'tracked':[]};images={};runtimes={}
        mode='baseline'
        @functools.wraps(original_step)
        def step(noise_pred,t,sample,*args,**kwargs):
            result=original_step(noise_pred,t,sample,*args,**kwargs);latents[mode].append(result[0].float().cpu())
            if mode=='tracked':
                r=tracker.read();maps.append(r.pop('maps'));flows.append(r.pop('flow'));frames.append({'step':len(frames)+1,'timestep':int(t),**r})
            return result
        model.noise_scheduler.step=step
        for mode in ['baseline','tracked']:
            tracker.enabled=(mode=='tracked');torch.manual_seed(42);g=torch.Generator('cuda').manual_seed(42);beg=time.time()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                im=pipe(**batch,num_inference_steps=20,guidance_scale=2.5,generator=g,ref_acceleration=False,repaint=False)[0][0]
            torch.cuda.synchronize();runtimes[mode]=time.time()-beg;images[mode]=np.asarray(im);im.save(dest/(mode+'.png'))
        parity={'max_latent_difference':max(float((x-y).abs().max()) for x,y in zip(latents['baseline'],latents['tracked'])),
                'max_image_difference':int(np.abs(images['baseline'].astype(int)-images['tracked'].astype(int)).max()),'steps':len(frames)}
        assert parity['max_latent_difference']==parity['max_image_difference']==0 and parity['steps']==20
        tracker.close();model.noise_scheduler.step=original_step
        np.savez_compressed(dest/'maps.npz',maps=np.stack(maps),flow=np.stack(flows))
        record={'case':case,'frames':frames,'parity':parity,'seconds':runtimes,'complete':True,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
        atomic(dest/'result.json',record);atomic(root/'PROGRESS.json',{'done':ci+1,'total':len(spec['cases']),'last':case['id'],'seconds':time.time()-start})
        print('CASE_COMPLETE',case['id'],len(case['points']),json.dumps(runtimes),flush=True)
    atomic(root/'COMPLETE.json',{'cases':len(spec['cases']),'seconds':time.time()-start,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'training':False})
    print('ALL_COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--setup-only',action='store_true');a=p.parse_args()
    if a.setup_only:print(json.dumps({'cases':[c['id'] for c in setup(a.root)['cases']]}))
    else:run(a.root)
