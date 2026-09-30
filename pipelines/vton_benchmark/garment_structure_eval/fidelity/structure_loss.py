"""Frozen full-image DINO + existing StructureExperts, with live Molmo points."""
import json
import time
import uuid
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .detail_protocol import atomic_json


def point_tensors(records, names, max_points, device):
    pts=torch.zeros(len(records),len(names),max_points,2,device=device)
    mask=torch.zeros(len(records),len(names),max_points,dtype=torch.bool,device=device)
    for b,record in enumerate(records):
        for k,name in enumerate(names):
            values=record.get(name,[])[:max_points]
            for i,xy in enumerate(values):
                if len(xy)!=2 or not all(0<=float(v)<=1 for v in xy):
                    raise ValueError('Invalid Molmo coordinates')
                pts[b,k,i]=torch.tensor(xy,device=device);mask[b,k,i]=True
    return pts,mask


class StructureTeacher(nn.Module):
    def __init__(self, checkpoint, device='cuda'):
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor
        from garment_structure_eval.experts import StructureExperts
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.names=ck['concepts'];self.meta=ck['meta'];self.max_points=ck['cfg']['max_points']
        self.net=StructureExperts(len(self.names),d_in=self.meta['D'],grid=self.meta['grid'],max_points=self.max_points)
        self.net.load_state_dict(ck['state'],strict=True)
        self.backbone=AutoModel.from_pretrained(self.meta['model'],local_files_only=True)
        proc=AutoImageProcessor.from_pretrained(self.meta['model'],local_files_only=True)
        self.register_buffer('mean',torch.tensor(proc.image_mean).view(1,3,1,1))
        self.register_buffer('std',torch.tensor(proc.image_std).view(1,3,1,1))
        self.requires_grad_(False);self.to(device);self.eval()

    def train(self, mode=True):
        super().train(False);return self

    def forward(self, images, points, masks):
        # Preserve the checkpoint's full-image 224x224, 16x16 DINO feature grid.
        # No center crop: normalized Molmo coordinates keep their geometry.
        x=F.interpolate((images.float()+1)/2,size=(self.meta['input'],)*2,
                        mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        with torch.autocast(images.device.type,enabled=False):
            h=self.backbone(pixel_values=(x-self.mean)/self.std).last_hidden_state
            patch=h[:,1+self.meta['register_tokens']:]
            if patch.shape[1]!=self.meta['grid']**2:raise ValueError('DINO grid mismatch')
            return self.net(h[:,0],patch,points,masks)


def paired_distance(teacher, generated, target):
    ga,la,va=generated;gb,lb,vb=target
    local=torch.linalg.vector_norm(la-lb,dim=-1)
    valid=va & vb
    global_dist=torch.linalg.vector_norm(ga-gb,dim=-1)
    distances=torch.cat([global_dist[:,None],local],1)
    validity=torch.cat([torch.ones_like(global_dist[:,None],dtype=torch.bool),valid],1)
    weights=teacher.net.w.softmax(0)[None]*validity
    holistic=(weights*distances).sum(1)/weights.sum(1).clamp_min(1e-8)
    # Per-part term prevents learned global weights from drowning out details.
    per_part=(local*valid).sum(1)/valid.sum(1).clamp_min(1)
    return (per_part+holistic).mean(),dict(part_count=int(valid.sum()),
        per_part=float(per_part.mean().detach()),holistic=float(holistic.mean().detach()))


class MolmoClient:
    def __init__(self, job):self.job=Path(job)
    def locate(self, pred_path, target_path, parse_path, iid):
        ticket=uuid.uuid4().hex
        atomic_json(self.job/'requests'/f'{ticket}.json',dict(id=iid,prediction=str(pred_path),target=str(target_path),parse=str(parse_path)))
        result=self.job/'responses'/f'{ticket}.json';start=time.monotonic()
        while not result.exists():
            if (self.job/'molmo_failed.json').exists():raise RuntimeError((self.job/'molmo_failed.json').read_text())
            if time.monotonic()-start>900:raise TimeoutError('Molmo localization timed out')
            time.sleep(.2)
        record=json.loads(result.read_text())
        if 'error' in record:raise RuntimeError(record['error'])
        return record


class StructureSupervision:
    def __init__(self, teacher, job, rows):
        self.teacher=teacher;self.job=Path(job);self.client=MolmoClient(job)
        self.rows={r['id']:r for r in rows};self.cache={};self.calls=0
    def __call__(self,pred,target,ids):
        from .detail_runner import png
        if len(ids)!=1:raise ValueError('Live Molmo supervision currently requires batch size one')
        iid=ids[0];self.calls+=1
        path=self.job/'requests'/f'image_{uuid.uuid4().hex}_{iid}.png';png(pred[0],path)
        record=self.client.locate(path,self.rows[iid]['target'],self.rows[iid]['parse'],iid)
        gt=record['target'];pr=record['prediction'];fallback=[]
        # GT presence controls supervision; missing generated points cannot mask the error.
        for name in self.teacher.names:
            if gt.get(name) and not pr.get(name):pr[name]=gt[name];fallback.append(name)
        tp,tm=point_tensors([gt],self.teacher.names,self.teacher.max_points,pred.device)
        pp,pm=point_tensors([pr],self.teacher.names,self.teacher.max_points,pred.device)
        if iid not in self.cache:
            with torch.no_grad():self.cache[iid]=tuple(v.detach() for v in self.teacher(target,tp,tm))
        loss,metrics=paired_distance(self.teacher,self.teacher(pred,pp,pm),self.cache[iid])
        metrics.update(fallback_parts=fallback,localization_request=record['request'],
                       target_parts=int(tm.any(-1).sum()),generated_parts=record['generated_parts'])
        return loss,metrics
