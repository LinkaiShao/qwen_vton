"""CPU-only checks using the actual pinned LeFFA processor and model configs."""
import argparse
import json
from pathlib import Path
import sys
import torch
from core import NativeAttentionTracker,LAYERS,noise_weight
from prepare import atomic


def main():
    p=argparse.ArgumentParser()
    for name in ['leffa-code','leffa-weights','output']:p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    sys.path.insert(0,str(a.leffa_code))
    from leffa.model import AttnProcessor2_0,LeffaModel
    from diffusers.models.attention_processor import Attention
    from diffusers import DDPMScheduler
    layer=Attention(query_dim=8,heads=2,dim_head=4,processor=AttnProcessor2_0()).eval()
    model=torch.nn.ModuleDict({'a':layer});x=torch.randn(1,24,8)
    original=model['a'](x)
    tracker=NativeAttentionTracker(model,layers=['a'],image_size=(4,3),output_grid=(4,3))
    observed=model['a'](x)
    assert torch.equal(original,observed)
    projected_q=layer.to_q(x).reshape(1,24,2,4).transpose(1,2)[:,:,:12]
    projected_k=layer.to_k(x).reshape(1,24,2,4).transpose(1,2)
    reference=(projected_q@projected_k.transpose(-1,-2)/2).softmax(-1)[...,12:].mean(1)
    torch.testing.assert_close(tracker.mean(),reference)
    tracker.clear();model['a'](x).square().mean().backward()
    norms={name:float(getattr(layer,name).weight.grad.norm()) for name in ['to_q','to_k','to_v']}
    assert all(v>0 for v in norms.values())
    tracker.close()
    with torch.device('meta'):
        full=LeffaModel(str(a.leffa_weights/'stable-diffusion-inpainting'),dtype='float32',height=512,width=384)
    scheduler=DDPMScheduler.from_pretrained(a.leffa_weights/'stable-diffusion-inpainting',subfolder='scheduler')
    assert not scheduler.alphas_cumprod.is_meta
    assert scheduler.config.prediction_type=='epsilon'
    w=noise_weight(scheduler.alphas_cumprod)
    assert len(w)==1000 and ((w>0)&(w<=1)).all()
    for name in LAYERS:assert full.unet.get_submodule(name).to_q is not None
    parameters=0;names=[]
    for name,mod in full.unet.named_modules():
        if name.startswith(('up_blocks.2.','up_blocks.3.')) and name.endswith('.attn1'):
            for field in ['to_q','to_k','to_v','to_out.0']:
                parameters+=sum(p.numel() for p in mod.get_submodule(field).parameters());names.append(name+'.'+field)
    result={'passed':True,'actual_processor_output_unchanged':True,'actual_probabilities_match':True,
            'processor_generation_gradient_norms':norms,'trainable_parameters':parameters,
            'native_projection_modules':names,'scheduler':'epsilon','all_timestep_weights_positive':True,
            'minimum_timestep_weight':float(w.min()),'gpu_used':False,
            'limitation':'CPU architecture checks; not real-image H200 forward/backward verification.'}
    atomic(a.output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
