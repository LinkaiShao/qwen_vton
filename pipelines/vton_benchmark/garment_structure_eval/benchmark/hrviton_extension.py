"""Explicit-flow HR-VITON extension; preserve the original eight-model run."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from .common import HERE, MODELS, EXPERT_SHA, atomic, digest, read, rows


def cohort(source):
    ids = {p.stem for p in (source/'collected/scores/leffa').glob('*.json')}
    assert len(ids) == 128
    for key in MODELS:
        assert {p.stem for p in (source/'collected/scores'/key).glob('*.json')} == ids
    records = [r for r in rows(source/'collected') if r['id'] in ids]
    assert len(records) == 128 and len({r['id'] for r in records}) == 128
    return records


def prepare(a):
    records = cohort(a.source)
    a.job.mkdir(parents=True, exist_ok=True)
    (a.job/'data').mkdir(exist_ok=True)
    if not (a.job/'data/test').exists():
        (a.job/'data/test').symlink_to(a.dataset/'test', target_is_directory=True)
    (a.job/'data/cohort.txt').write_text(''.join(f"{r['id']}.jpg {r['id']}.jpg\n" for r in records))
    (a.job/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    from .prepare import mask_from_dataset
    for r in records:
        for field, expected in r['hashes'].items():
            assert digest(a.dataset/r[field]) == expected, (r['id'], field)
        p = a.job/'masks'/(r['id']+'.png')
        if not p.exists():
            p.parent.mkdir(exist_ok=True)
            mask_from_dataset(a.dataset, r['id']).save(p)
    repo = HERE/'vendor/hrviton'
    atomic(a.job/'provenance.json', dict(model='HR-VITON', inference_only=True,
        repository='https://github.com/sangyun884/HR-VITON',
        revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        weights={name:dict(sha256=digest(a.job/'checkpoints'/name), google_drive_id=gid)
                 for name,gid in [('condition.pth','1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ'),
                                  ('generator.pth','1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy')]},
        cohort_sha256=digest(a.job/'manifest.jsonl'), source=str(a.source), samples=128,
        primary='Shared benchmark agnostic mask, gray 127 fill; native parse-agnostic and DensePose conditions',
        secondary='Official native HR-VITON agnostic construction; identical cases and seeds',
        size=[1024,768],dtype='float32',occlusion=True,clothmask_composition='warp_grad',
        sampling='Official grid_sample default align_corners=False; no geometry changes',
        scoring='Primary fixed-GT-point frozen expert distance only; reuse original Molmo GT coordinates',
        preprocessing_note='Required native parse-agnostic and cloth-mask conditions retained; no extra final RGB compositing'))
    print('PREPARED',len(records),flush=True)


def infer(a):
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn
    from torch.nn import functional as F
    import torchgeometry as tgm
    sys.path.insert(0,str(HERE/'vendor/hrviton'))
    from cp_dataset_test import CPDatasetTest
    from networks import ConditionGenerator, make_grid
    from network_generator import SPADEGenerator
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(20260930)
    opt=SimpleNamespace(cuda=True, dataroot=str(a.job/'data'), datamode='test',
        data_list='cohort.txt', fine_width=768, fine_height=1024, semantic_nc=13,
        output_nc=13, gen_semantic_nc=7, warp_feature='T1', out_layer='relu',
        norm_G='spectralaliasinstance', ngf=64, init_type='xavier', init_variance=.02,
        num_upsampling_layers='most')
    dataset=CPDatasetTest(opt)
    # The official main mutates this option to 7 after constructing the dataset.
    # Keep a separate generator option so the dataset retains 13 input classes.
    tocg=ConditionGenerator(opt,4,16,13,ngf=96,norm_layer=nn.BatchNorm2d)
    gen_opt=SimpleNamespace(**{**vars(opt),'semantic_nc':7})
    generator=SPADEGenerator(gen_opt,9)
    tocg.load_state_dict(torch.load(a.job/'checkpoints/condition.pth',map_location='cpu',weights_only=False),strict=True)
    state=torch.load(a.job/'checkpoints/generator.pth',map_location='cpu',weights_only=False)
    remap=lambda k:k.replace('ace','alias').replace('.Spade','')
    mapped=OrderedDict((remap(k),v) for k,v in state.items())
    if hasattr(state,'_metadata'):mapped._metadata=OrderedDict((remap(k),v) for k,v in state._metadata.items())
    generator.load_state_dict(mapped,strict=True)
    tocg.cuda().eval();generator.cuda().eval()
    gauss=tgm.image.GaussianBlur((15,15),(3,3)).cuda()
    def save(x,path,mask=False):
        path.parent.mkdir(parents=True,exist_ok=True)
        arr=x.detach().float().cpu()[0]
        arr=(arr if mask else (arr+1)/2).clamp(0,1)
        arr=(arr*255).round().byte().permute(1,2,0).numpy()
        Image.fromarray(arr[:,:,0] if arr.shape[-1]==1 else arr).save(path)
    records=rows(a.job)[:a.limit]
    begin=time.monotonic()
    with torch.inference_mode():
        for idx,r in enumerate(records):
            iid=r['id'];receipt=a.job/'inference'/f'{iid}.json'
            if receipt.exists():
                old=read(receipt)
                assert all(digest(a.job/k)==v for k,v in old['artifacts'].items())
                continue
            inputs=dataset[idx]
            assert inputs['im_name']==iid+'.jpg'
            cloth=inputs['cloth']['paired'][None].cuda()
            cm=(inputs['cloth_mask']['paired'][None]>.5).float().cuda()
            pa=inputs['parse_agnostic'][None].cuda();dp=inputs['densepose'][None].cuda()
            small=lambda x,mode:F.interpolate(x,size=(256,192),mode=mode)
            torch.cuda.synchronize();start=time.monotonic()
            flows,seg,_,warped_cm_small=tocg(opt,torch.cat([small(cloth,'bilinear'),small(cm,'nearest')],1),
                torch.cat([small(pa,'nearest'),small(dp,'bilinear')],1))
            composition=torch.ones_like(seg);composition[:,3:4]=warped_cm_small
            seg=seg*composition
            logits=gauss(F.interpolate(seg,size=(1024,768),mode='bilinear'))
            labels=logits.argmax(1)[:,None]
            old_parse=torch.zeros(1,13,1024,768,device='cuda').scatter_(1,labels,1.)
            groups=[[0],[2,4,7,8,9,10,11],[3],[1],[5],[6],[12]]
            parse=torch.cat([old_parse[:,g].sum(1,keepdim=True) for g in groups],1)
            flow=F.interpolate(flows[-1].permute(0,3,1,2),size=(1024,768),mode='bilinear').permute(0,2,3,1)
            norm=torch.cat([flow[:,:,:,0:1]/((96-1.)/2.),flow[:,:,:,1:2]/((128-1.)/2.)],3)
            sampling_grid=make_grid(1,1024,768,opt)+norm
            raw_warp=F.grid_sample(cloth,sampling_grid,padding_mode='border',align_corners=False)
            warped_mask=F.grid_sample(cm,sampling_grid,padding_mode='border',align_corners=False)
            probabilities=F.softmax(logits,dim=1)
            overlap=torch.cat([probabilities[:,1:3],probabilities[:,5:]],1).sum(1,keepdim=True)
            warped_mask=warped_mask-overlap*warped_mask
            warp=raw_warp*warped_mask+torch.ones_like(raw_warp)*(1-warped_mask)
            torch.cuda.synchronize();warp_s=time.monotonic()-start
            artifacts=[]
            for label,x,is_mask in [('raw_warp',raw_warp,False),('warp',warp,False),('warped_mask',warped_mask,True)]:
                p=a.job/label/f'{iid}.png';save(x,p,is_mask);artifacts.append(p)
            p=a.job/'flows'/f'{iid}.npz';p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,flow=flows[-1][0].cpu().numpy(),output_hw=[1024,768],align_corners=False)
            artifacts.append(p)
            mask=np.asarray(Image.open(a.job/'masks'/f'{iid}.png').convert('L')).copy()
            mask=torch.from_numpy(mask)[None,None].cuda().float()/255
            native=inputs['agnostic'][None].cuda()
            shared=inputs['image'][None].cuda()*(1-mask)+(127/127.5-1)*mask
            timings={}
            for mode,agnostic in [('hrviton',shared),('hrviton_native',native)]:
                torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
                torch.cuda.synchronize();start=time.monotonic()
                output=generator(torch.cat([agnostic,dp,warp],1),parse)
                torch.cuda.synchronize();timings[mode]=time.monotonic()-start
                assert output.shape==(1,3,1024,768) and torch.isfinite(output).all()
                p=a.job/'predictions'/mode/f'{iid}.png';save(output,p);artifacts.append(p)
                p=a.job/'agnostic'/mode/f'{iid}.png';save(agnostic,p);artifacts.append(p)
                del output
            atomic(receipt,dict(id=iid,seed=r['seed'],gpu=torch.cuda.get_device_name(),
                warp_seconds=warp_s,generator_seconds=timings,mask_sha256=digest(a.job/'masks'/f'{iid}.png'),
                cloth_mask_sha256=digest(a.dataset/'test/cloth-mask'/f'{iid}.jpg'),
                parse_agnostic_sha256=digest(a.dataset/'test/image-parse-agnostic-v3.2'/f'{iid}.png'),
                artifacts={str(p.relative_to(a.job)):digest(p) for p in artifacts},
                max_allocated_bytes=torch.cuda.max_memory_allocated()))
            print('GENERATED',idx+1,len(records),iid,'seconds',round(warp_s+sum(timings.values()),2),flush=True)
    atomic(a.job/'inference_status.json',dict(complete=len(list((a.job/'inference').glob('*.json')))==128,
        cases=len(list((a.job/'inference').glob('*.json'))),elapsed_seconds=time.monotonic()-begin,
        torch=torch.__version__,gpu=torch.cuda.get_device_name()))


def score(a):
    import numpy as np
    from PIL import Image
    import torch
    from garment_structure_eval.fidelity.structure_loss import StructureTeacher,point_tensors
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    expert=a.source/'experts.pt'
    assert digest(expert)==EXPERT_SHA
    teacher=StructureTeacher(str(expert)).eval()
    def embed(path,points):
        x=torch.from_numpy(np.array(Image.open(path).convert('RGB')).copy()).permute(2,0,1)[None].cuda().float()/127.5-1
        p,m=point_tensors([points],teacher.names,teacher.max_points,'cuda')
        return teacher(x,p,m)
    def distances(x,y):return torch.linalg.vector_norm(x[1]-y[1],dim=-1)[0]
    checks=[]
    with torch.inference_mode():
        for idx,r in enumerate(rows(a.job)[:a.limit]):
            old=read(a.source/'collected/scores/leffa'/f"{r['id']}.json")
            gtfile=a.source/'collected'/old['gt_grounding']
            assert digest(gtfile)==old['gt_grounding_sha256']
            points=read(gtfile)['points'];gt=a.dataset/r['target']
            assert digest(gt)==old['target_sha256']
            reference=embed(gt,points)
            if idx<4:
                control=a.source/'collected/predictions/leffa'/f"{r['id']}.png"
                assert digest(control)==old['prediction_sha256']
                d=distances(embed(control,points),reference)
                error=max(abs(float(d[k])-p['fixed_distance']) for k,p in enumerate(old['parts']) if p['gt_present'])
                self_error=float(distances(embed(gt,points),reference).max())
                assert error<1e-4 and self_error<1e-5,(error,self_error)
                checks.append(dict(id=r['id'],prior_score_max_error=error,gt_self_error=self_error))
            for mode in ['hrviton','hrviton_native']:
                image=a.job/'predictions'/mode/f"{r['id']}.png"
                if not image.exists():continue
                out=a.job/'scores'/mode/f"{r['id']}.json"
                if out.exists():assert read(out)['prediction_sha256']==digest(image);continue
                pred=embed(image,points);d=distances(pred,reference)
                assert torch.isfinite(d).all()
                parts=[dict(concept=p['concept'],gt_present=p['gt_present'],gt_status=p['gt_status'],
                    fixed_distance=float(d[k]) if p['gt_present'] else None,gt_points=p['gt_points'])
                    for k,p in enumerate(old['parts'])]
                atomic(out,dict(id=r['id'],model=mode,seed=r['seed'],prediction_sha256=digest(image),
                    target_sha256=digest(gt),expert_sha256=EXPERT_SHA,gt_grounding_sha256=digest(gtfile),
                    grounding_protocol=old['grounding_protocol'],prediction_localization='not evaluated; fixed GT locations only',
                    global_distance=float(torch.linalg.vector_norm(pred[0]-reference[0])),parts=parts))
            print('SCORED',idx+1,r['id'],flush=True)
    atomic(a.job/'scorer_validation.json',dict(passed=True,checks=checks,expert_sha256=EXPERT_SHA,
        frozen=all(not p.requires_grad for p in teacher.parameters()),torch=torch.__version__))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','score'])
    p.add_argument('--job',type=Path,default=Path('/mnt/nvme0/vton_region_benchmark/20260930_hrviton'))
    p.add_argument('--source',type=Path,default=Path('/mnt/nvme0/vton_region_benchmark/20260925_eight_models_v2'))
    p.add_argument('--dataset',type=Path,default=Path('VITON-HD-dataset'))
    p.add_argument('--limit',type=int,default=128)
    a=p.parse_args();a.dataset=a.dataset.resolve();a.job=a.job.resolve();a.source=a.source.resolve()
    globals()[a.stage](a)
