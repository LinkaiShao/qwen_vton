"""Post-training check: matched full samples, never used inside the training loss.

Uses the official LeFFA inference pipeline with 20 DDPM steps and CFG=2.5, all
eight development garments, fixed per-image seeds, and no final RGB repaint.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from verify_vton_localization import Experiment, image_tensor, atomic, CONCEPTS


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data','run','output','leffa-code','leffa-weights','dino-weights']:
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Use a new output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    recorded=json.loads((a.run/'RUN.json').read_text())['arguments']
    for name in ['height','width','aggregation','pointed_mix']:
        setattr(a,name,recorded.get(name, {'aggregation':'mean_probs','pointed_mix':.5}.get(name)))
    torch.set_num_threads(4);torch.manual_seed(20260928)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():
        raise RuntimeError('H200 required')
    manifest=json.loads((a.data/'manifest.json').read_text())
    experiment=Experiment(a,manifest)
    experiment.tracker.enabled=False
    sys.path.insert(0,str(a.leffa_code))
    from leffa.pipeline import LeffaPipeline
    pipeline=LeffaPipeline(experiment.model)
    records=[]
    for arm in ['baseline','denoise_spatial_control','denoise_spatial_dino']:
        experiment.reset()
        if arm!='baseline':
            ckpt=torch.load(a.run/(arm+'_latest.pt'),map_location='cpu',weights_only=True)
            for param,value in zip(experiment.projections,ckpt['projection_parameters']):
                param.copy_(value.to(param.device))
        for row in experiment.val_rows:
            iid=row['row']['id'];folder=a.data/'data'/iid
            seed=int(hashlib.sha256(('fullsample:'+iid).encode()).hexdigest()[:8],16)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            images={k:image_tensor(folder/(k+'.png')).cuda() for k in ['target','garment','densepose']}
            mask=image_tensor(folder/'mask.png',mask=True).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                image=pipeline(src_image=images['target'],ref_image=images['garment'],
                               mask=mask,densepose=images['densepose'],num_inference_steps=20,
                               do_classifier_free_guidance=True,guidance_scale=2.5,
                               generator=torch.Generator(device='cuda').manual_seed(seed),
                               repaint=False)[0][0]
            dest=a.output/arm/(iid+'.png');dest.parent.mkdir(exist_ok=True);image.save(dest)
            tensor=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].float().cuda()/127.5-1
            distance=experiment.evaluator.distances(tensor,row['boxes'],row['target_features'])
            for index,concept in enumerate(CONCEPTS):
                if row['valid'][0,index]:
                    records.append({'arm':arm,'id':iid,'concept':concept,'fixed_roi_dino_distance':float(distance[index]),'seed':seed})
            print('FULL_SAMPLE',arm,iid,flush=True)
    result={'purpose':'Post-training evaluation only. No multi-step sampling was used for training losses.',
            'steps':20,'guidance_scale':2.5,'repaint':False,'resolution':[a.height,a.width],
            'cohort':'Same eight development garments; not an untouched final test set.',
            'records':records,'mean_fixed_roi_dino_distance':{arm:float(np.mean([r['fixed_roi_dino_distance'] for r in records if r['arm']==arm])) for arm in ['baseline','denoise_spatial_control','denoise_spatial_dino']}}
    atomic(a.output/'RESULT.json',result);print('SAMPLING_EVALUATION_COMPLETE',json.dumps(result['mean_fixed_roi_dino_distance']),flush=True)


if __name__=='__main__':
    main()
