"""Source-detail footprints from the exact generation Q/K. No learned readout."""
import math
import torch
from torch.nn import functional as F

LAYERS=tuple(f'up_blocks.{b}.attentions.{a}.transformer_blocks.0.attn1' for b in (2,3) for a in range(3))

def grid(n,ratio=4/3):
    h=round(math.sqrt(n*ratio));w=n//h
    if h*w!=n:raise ValueError('Unexpected person/reference spatial layout')
    return h,w

def part_attention(q,k,heads,source_parts,output_grid=(64,48),chunk=128):
    """B,H,parts,Hout,Wout; full-key softmax before reference slicing."""
    b,total,d=q.shape;n=total//2;h,w=grid(n)
    if total!=2*h*w or k.shape!=q.shape:raise ValueError('Unexpected concatenation')
    q=q[:,:n].reshape(b,n,heads,d//heads).transpose(1,2)
    k=k.reshape(b,total,heads,d//heads).transpose(1,2)
    masks=F.interpolate(source_parts.float(),(h,w),mode='area').flatten(2)
    pieces=[]
    with torch.autocast(q.device.type,enabled=False):
        for start in range(0,n,chunk):
            p=(q[:,:,start:start+chunk].float()@k.float().transpose(-1,-2)/math.sqrt(d//heads)).softmax(-1)
            pieces.append(p[...,n:]@masks[:,None].transpose(-1,-2))
    maps=torch.cat(pieces,2).permute(0,1,3,2).reshape(b*heads,-1,h,w)
    maps=F.interpolate(maps,output_grid,mode='bilinear',align_corners=False)
    return maps.reshape(b,heads,-1,*output_grid)

class DetailTracker:
    def __init__(self,unet,layers=LAYERS,output_grid=(64,48)):
        self.layers=tuple(layers);self.output_grid=output_grid
        self.enabled=True;self.parts=None;self.maps={};self.native_grids={};self.captured={};self.handles=[];self.intervention=None
        for name in self.layers:
            module=unet.get_submodule(name)
            for kind in ('q','k'):
                def capture(mod,inp,out,name=name,kind=kind):
                    if self.enabled:self.captured[(name,kind)]=out
                self.handles.append(getattr(module,'to_'+kind).register_forward_hook(capture))
            def values(mod,inp,out,name=name,heads=module.heads):
                spec=self.intervention
                if spec is None or name not in spec['layers']:return
                if torch.is_grad_enabled():raise RuntimeError('Audit interventions are inference-only')
                n=out.shape[1]//2;h,w=grid(n)
                mask=F.interpolate(spec['mask'].float(),(h,w),mode='area').flatten(2)[0,0]
                value=out.clone();v=value[:,n:].view(out.shape[0],n,heads,-1)
                scale=1+(spec['factor']-1)*mask
                if spec['head'] is None:v.mul_(scale[None,:,None,None])
                else:v[:,:,spec['head']].mul_(scale[None,:,None])
                return value
            self.handles.append(module.to_v.register_forward_hook(values))
            def finish(mod,inp,out,name=name):
                if not self.enabled:return
                if self.parts is None:raise RuntimeError('Source-only part regions must be provided')
                q=self.captured.pop((name,'q'));k=self.captured.pop((name,'k'))
                self.native_grids[name]=grid(q.shape[1]//2)
                self.maps[name]=part_attention(q,k,mod.heads,self.parts,self.output_grid)
            self.handles.append(module.register_forward_hook(finish))

    def clear(self):self.maps.clear();self.captured.clear()
    def candidates(self):
        maps=[];names=[]
        for name in self.layers:
            h=self.maps[name][0]
            for j in range(len(h)):maps.append(h[j]);names.append(name+f'/head{j}')
            maps.append(h.mean(0));names.append(name+'/mean')
        maps.append(torch.stack([self.maps[n][0].mean(0) for n in self.layers]).mean(0));names.append('all/mean')
        return torch.stack(maps),names

    def set_intervention(self,candidate,mask,factor):
        if candidate=='all/mean':layers=self.layers;head=None
        else:
            name,head=candidate.rsplit('/',1);layers=(name,);head=None if head=='mean' else int(head[4:])
        self.intervention={'layers':layers,'head':head,'mask':mask,'factor':factor}

    def close(self):
        for h in self.handles:h.remove()
        self.clear()
