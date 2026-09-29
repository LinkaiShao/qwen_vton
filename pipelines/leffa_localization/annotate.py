"""Offline SAM3 detail regions. Never called inside the LeFFA forward/readout."""
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import label
from common import *

# Frozen after checking development images, before any heldout predictions.
# Qwen is used ONLY for the reference garment noun, never for coordinates.
PROMPTS={'collar':('collar','shirt collar'), 'neckline':('neckline','shirt neckline'),
         'cuff':('cuff','sleeve cuff'), 'sleeve_opening':('sleeve hem','end of sleeve')}

def box_masks(annotation,identity):
    """Compatibility utility for explicit normalized boxes; not the mask teacher."""
    masks=[];parts={};h,w=identity.shape
    for name in PARTS:
        rec=dict(annotation['parts'][name]);mask=np.zeros((h,w),bool)
        for x1,y1,x2,y2 in rec['boxes']:
            mask[int(y1*h):int(np.ceil(y2*h)),int(x1*w):int(np.ceil(x2*w))]=True
        mask &= identity
        if rec['status']=='present' and not mask.any():rec['status']='uncertain'
        masks.append(mask);parts[name]=rec
    return np.stack(masks),parts

def agreement_masks(first,second,identity):
    """Accept spatial agreement, never invent a location or fill garment holes."""
    out=np.zeros_like(identity);matches=[]
    for a in first:
        for b in second:
            inter=a&b;union=a|b;iou=float(inter.sum()/max(1,union.sum()))
            if iou<.35 or inter.sum()<8:continue
            overlap=float((inter&identity).sum()/max(1,inter.sum()))
            if overlap<.60 or (inter&identity).sum()>.35*identity.sum():continue
            out |= inter&identity;matches.append({'iou':iou,'identity_overlap':overlap})
    return out,matches

def mask_boxes(mask):
    cc,n=label(mask);boxes=[];h,w=mask.shape
    for j in range(1,n+1):
        y,x=np.where(cc==j)
        if len(x)>=8:boxes.append([float(x.min()/w),float(y.min()/h),float((x.max()+1)/w),float((y.max()+1)/h)])
    return boxes

def instance_masks(masks):
    """Keep sleeve ends separate; a shared enclosing box would include the torso."""
    out=[];names=[]
    for j,name in enumerate(PARTS):
        if name not in ('cuff','sleeve_opening'):continue
        cc,n=label(masks[j]);regions=[cc==i for i in range(1,n+1) if (cc==i).sum()>=8]
        regions=sorted(regions,key=lambda m:-m.sum())[:2]
        regions.sort(key=lambda m:np.where(m)[1].mean())
        for i,m in enumerate(regions):out.append(m);names.append(name+f'_source_instance_{i+1}')
    return np.stack(out) if out else np.empty((0,*masks.shape[-2:]),bool),names

def vision(cutter,im):
    import torch
    x=cutter.proc(images=im,return_tensors='pt').to('cuda')
    with torch.inference_mode():return cutter.net.get_vision_features(pixel_values=x['pixel_values'].to(torch.bfloat16))

def detail_regions(cutter,im,identity,queries):
    v=vision(cutter,im);masks=[];parts={}
    for name in PARTS:
        proposals=[];evidence=[]
        for prompt in PROMPTS[name]:
            result,ev=cutter.predict(v,prompt,im.size,.35)
            proposals.append([m.cpu().numpy().astype(bool) for m in result['masks']]);evidence.append(ev)
        mask,agreements=agreement_masks(*proposals,identity)
        # A missing detection is NOT proof of absence. All unresolved parts are uncertain.
        status='present' if mask.any() else 'uncertain'
        parts[name]={'status':status,'boxes':mask_boxes(mask),'annotation_kind':'sam3_two_prompt_consensus',
                     'evidence':evidence,'agreements':agreements,'pixels':int(mask.sum())}
        masks.append(mask)
    return np.stack(masks),{'parts':parts,'queries':queries,'coordinate_system':'normalized_original_image',
        'teacher':'facebook/sam3@3c879f39826c281e95690f02c7821c4de09afae7',
        'no_detection_means':'uncertain, not absent','mask_coordinates_from':'native SAM3 output; no VLM coordinate conversion'}

def names(root,limit=None):
    bind_gpu();sys.path.insert(0,str(cutter_path()))
    from flatlay_cutter.setup import SetupModel,parse_object,specific_names
    from flatlay_cutter.prompts import NAMING_PROMPT
    model=SetupModel(root/'annotation_cache',local_files_only=True)
    rows=sorted(manifest(root),key=lambda r:(r['role']!='regression',r['key']))
    if limit:rows=rows[:limit]
    try:
        for n,row in enumerate(rows):
            key=row['key'];path=root/'data'/key/'garment.png';out=root/'annotations/names'/(key+'.json')
            if out.exists():continue
            response=model.ask([path],NAMING_PROMPT,64)
            queries=specific_names(parse_object(response['raw']).get('queries',[]))
            if not queries:raise ValueError('No specific reference noun: '+key)
            atomic(out,{'queries':queries,'source_sha256':digest(path),'evidence':response})
            atomic(root/'ANNOTATION_PROGRESS.json',{'view':'reference_names','done':n+1,'total':len(rows),'key':key})
            print('NAMED',n+1,len(rows),key,queries,flush=True)
    finally:model.close()

def generated(root,limit=None):
    bind_gpu();sys.path.insert(0,str(cutter_path()))
    from flatlay_cutter.engine import ReferenceCutter
    cutter=ReferenceCutter(local_files_only=True);jobs=[]
    for row in manifest(root):
        for t in TIMESTEPS:
            path=root/'probe'/row['key']/f'{t}.png'
            if path.exists():jobs.append((row['key'],t,path))
    if limit:jobs=jobs[:limit]
    for n,(key,t,path) in enumerate(jobs):
        out=root/'annotations/generated'/key/f'{t}.json'
        if out.exists():continue
        im=Image.open(path).convert('RGB');queries=read(root/'annotations/names'/(key+'.json'))['queries']
        # Crucially no target/GT identity clips generated-image detail detections.
        masks,record=detail_regions(cutter,im,np.ones(im.size[::-1],bool),queries)
        out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),masks=masks)
        atomic(out,{**record,'key':key,'t':t,'view':'generated','source_sha256':digest(path)})
        atomic(root/'ANNOTATION_PROGRESS.json',{'view':'generated','done':n+1,'total':len(jobs),'key':key})
        print('ANNOTATED generated',n+1,len(jobs),key,t,flush=True)

if __name__=='__main__':
    p=arguments(__doc__);p.add_argument('view',choices=['names','generated']);p.add_argument('--limit',type=int)
    a=p.parse_args();(names if a.view=='names' else generated)(a.root,a.limit)
