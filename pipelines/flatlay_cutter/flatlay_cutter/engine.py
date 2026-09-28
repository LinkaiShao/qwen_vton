"""Flatlay-conditioned SAM3 selection with bounded prompt recovery.

No titles, annotations, other worn views, or per-SKU manual rules are inputs.
SAM3 boundaries remain intact: appearance scores select whole proposals.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
try:import pillow_avif
except ImportError:pass
from .base import DirectCutter, reference_palette, color_support
from .io import read_image, digest

def aliases(query):
    q=query.lower().strip()
    if q=='bodysuit':return ['T-shirt','top']
    if q in ['cargo pants','track pants','trousers','jeans']:return ['pants','trousers']
    if q in ['sweater','knitwear','jumper']:return ['turtleneck','top']
    if q in ['tank top','tank','sleeveless top','sleeveless shirt','bikini top']:return ['top','shirt']
    if q in ['shirt','polo shirt','t-shirt','sweatshirt','hoodie','blouse','cardigan']:return ['top','shirt']
    if 'jacket' in q or q in ['coat','blazer']:return ['jacket','coat']
    if 'bag' in q:return ['bag','handbag']
    if q in ['sneakers','sneaker','shoe','shoes']:return ['shoes','footwear']
    return []

def normalized_features(vision):
    f=vision.fpn_hidden_states[-2][0].float()
    return F.normalize(f.flatten(1).T,dim=-1).half(),f.shape[-2:]

def feature_mask(mask,shape):
    return torch.from_numpy(np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(shape[::-1],Image.Resampling.NEAREST)).copy()).to('cuda').bool().flatten()

def appearance_gate(records):
    """Compare candidates separately for every reference component, then retain their union."""
    valid=[r for r in records if r['color_support']>=.35 and r['feature_pixels']>=1]
    if not valid:return
    for bank in range(len(valid[0]['component_scores'])):
        best_pool=max(r['component_scores'][bank]['cosine'] for r in valid)
        compatible=[r for r in valid if r['component_scores'][bank]['cosine']>=.75*best_pool]
        best_margin=max((r['component_scores'][bank]['margin'] for r in compatible),default=0.)
        for r in compatible:
            margin=r['component_scores'][bank]['margin']
            if margin>0 and margin>=.6*best_margin:
                r['appearance_keep']=True;r['keep']=True;r['matched_reference_components'].append(bank)

class ReferenceCutter(DirectCutter):
    def __init__(self, **model_options):
        super().__init__(**model_options);self.text_cache={}

    def text(self,q):
        if q not in self.text_cache:
            inp=self.proc(text=q,return_tensors='pt').to('cuda')
            with torch.inference_mode():self.text_cache[q]=(self.net.get_text_features(**inp).pooler_output,inp['attention_mask'])
        return self.text_cache[q]

    def predict(self,vision,q,size,threshold,anchors=False):
        text,attention=self.text(q)
        with torch.inference_mode():out=self.net(vision_embeds=vision,text_embeds=text,attention_mask=attention)
        presence=float(out.presence_logits.sigmoid().flatten()[0]);instance=float(out.pred_logits.sigmoid().max())
        result=self.proc.post_process_instance_segmentation(out,threshold=threshold,mask_threshold=.5,target_sizes=[size[::-1]])[0]
        if anchors:
            raw=out.pred_masks[0][out.pred_logits[0].sigmoid()>=.5]
            shape=vision.fpn_hidden_states[-2].shape[-2:]
            result['raw_anchors']=F.interpolate(raw[:,None].float(),shape,mode='bilinear',align_corners=False)[:,0]>0 if len(raw) else torch.zeros((0,*shape),device='cuda',dtype=torch.bool)
        return result,{'query':q,'presence':presence,'max_instance_score':instance,'max_final_score':presence*instance,'threshold':threshold,'instances':len(result['scores'])}

    @torch.inference_mode()
    def prepare(self,sku,query_overrides=None):
        start=time.perf_counter();queries=[];images=[];banks=[];evidence=[]
        for ref in sku['references']:
            names=(query_overrides or {}).get(ref['id'],ref['reference_queries'])
            if any(q.lower().strip() in ['clothing','garment','clothes','apparel','fashion item'] for q in names):
                raise ValueError('Generic reference noun requires automatic reclassification: '+ref['id'])
            queries.extend(q for q in names if q not in queries);im=read_image(ref['source'])
            inp=self.proc(images=im,return_tensors='pt').to('cuda');v=self.net.get_vision_features(pixel_values=inp['pixel_values'].to(torch.bfloat16));feat,shape=normalized_features(v)
            old=Path(ref['mask']) if ref.get('mask') else None;attempts=[]
            if ref.get('source_sha256') and digest(ref['source'])!=ref['source_sha256']:
                raise ValueError('Reference source changed: '+ref['id'])
            if old is not None:
                if ref.get('mask_sha256') and digest(old)!=ref['mask_sha256']:
                    raise ValueError('Reference mask changed: '+ref['id'])
                mask=np.asarray(Image.open(old).convert('L'))>0
                if mask.shape!=im.size[::-1] or not mask.any():
                    raise ValueError('Reference mask must be nonempty and match EXIF-corrected source size')
            else:
                mask=np.zeros(im.size[::-1],bool)
                for original in names:
                    for q in dict.fromkeys([original]+aliases(original)):
                        result,ev=self.predict(v,q,im.size,.35);attempts.append(ev)
                        if len(result['masks']):mask|=result['masks'].any(0).cpu().numpy().astype(bool);break
                if not mask.any():raise ValueError('Reference segmentation exhausted bounded aliases: '+ref['id'])
            parts=[mask];accessory=all(any(t in q.lower() for t in ['bag','shoe','sneaker','footwear']) for q in names)
            if not accessory:
                # Broad prompting is safe only on the isolated product, never on a worn outfit.
                all_parts,part_ev=self.predict(v,'clothing',im.size,.35);attempts.append(dict(part_ev,role='reference_component_coverage'))
                for part in all_parts['masks']:
                    part=part.cpu().numpy().astype(bool)
                    if part.sum()<64 or any((part&p).sum()/max(1,(part|p).sum())>.95 for p in parts):continue
                    parts.append(part)
            union=np.logical_or.reduce(parts);fm=feature_mask(union,shape);bg=feat[~fm]
            if not len(bg):raise ValueError('Reference needs background feature support')
            bg=bg[::max(1,len(bg)//1024)];ids=[]
            for part in parts:
                fg=feat[feature_mask(part,shape)]
                if not len(fg):continue
                fg=fg[::max(1,len(fg)//1024)];ids.append(len(banks));banks.append((fg,bg,F.normalize(fg.float().mean(0),dim=0)))
            if not ids:raise ValueError('Reference has no foreground feature support')
            images.append((im,union));evidence.append({'id':ref['id'],'queries':names,'recovered_reference':old is None,'attempts':attempts,'source_sha256':digest(ref['source']),'component_ids':ids,'component_pixels':[int(p.sum()) for p in parts]})
        for q in queries:
            for name in [q]+aliases(q):self.text(name)
        self.refs[sku['key']]={'queries':queries,'palette':reference_palette(images),'banks':banks,'reference_ids':[r['id'] for r in sku['references']],'reference_evidence':evidence}
        torch.cuda.synchronize();return time.perf_counter()-start

    @torch.inference_mode()
    def cut(self,image,sku):
        ref=self.refs[sku];torch.cuda.synchronize();start=time.perf_counter();inp=self.proc(images=image,return_tensors='pt').to('cuda');vision=self.net.get_vision_features(pixel_values=inp['pixel_values'].to(torch.bfloat16));feat,shape=normalized_features(vision)
        masks=[];records=[];attempts=[]
        for primary in ref['queries']:
            anchors=None
            for attempt,q in enumerate(dict.fromkeys([primary]+aliases(primary))):
                result,ev=self.predict(vision,q,image.size,.20,anchors=attempt==0)
                if attempt==0:anchors=result['raw_anchors'];ev['strong_spatial_hypotheses']=len(anchors)
                ev.update(primary=primary,retry=attempt>0);attempts.append(ev);group=[];group_masks=[]
                for m,score in zip(result['masks'],result['scores']):
                    mask=m.cpu().numpy().astype(bool)
                    if mask.sum()<64:continue
                    fm=feature_mask(mask,shape);local=feat[fm];metrics=[]
                    if len(local):
                        proto=F.normalize(local.float().mean(0),dim=0)
                        for fg,bg,refproto in ref['banks']:
                            a=(local@fg.T).max(-1).values.float();b=(local@bg.T).max(-1).values.float()
                            metrics.append((float((a-b).mean()),float(proto@refproto),float(a.mean())))
                    margin,pool,fgsim=max(metrics,key=lambda x:x[0]) if metrics else (-1.,-1.,-1.)
                    anchor_iou=None
                    if attempt:
                        am=anchors.flatten(1);anchor_iou=float(((am&fm).sum(1)/(am|fm).sum(1).clamp_min(1)).max()) if len(am) else 0.
                    r={'query':q,'primary':primary,'score':float(score),'pixels':int(mask.sum()),'color_support':color_support(image,mask,ref['palette']),'feature_pixels':len(local),'appearance_margin':margin,'appearance_cosine':pool,'foreground_similarity':fgsim,'component_scores':[{'margin':a,'cosine':b,'foreground_similarity':c} for a,b,c in metrics],'matched_reference_components':[],'retry_anchor_iou':anchor_iou,'retry_anchor_keep':anchor_iou is None or anchor_iou>=.8,'appearance_keep':False,'keep':False}
                    group.append(r);group_masks.append(mask)
                appearance_gate(group)
                # A broader noun may rescue category confidence, but cannot select a new region.
                for r in group:r['keep']=r['keep'] and r['retry_anchor_keep']
                records.extend(group)
                chosen=[mask for mask,r in zip(group_masks,group) if r['keep']]
                if chosen:masks.extend(chosen);break
        union=np.logical_or.reduce(masks) if masks else np.zeros(image.size[::-1],bool);torch.cuda.synchronize()
        return union,{'seconds':time.perf_counter()-start,'reference_ids':ref['reference_ids'],'queries':ref['queries'],'candidates':records,'attempts':attempts,'reference_evidence':ref['reference_evidence'],'visual_encoder_passes':1,'per_target_vlm_calls':0}
