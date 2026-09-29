"""Reference-conditioned identity, protection, inpainting, and detail masks."""
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from common import *
from annotate import detail_regions

PROTECTED_LABELS=(1,2,3,4,8,9,12,13,14,15,16,17,18,19)

def construct_masks(selected,parse):
    protected=np.isin(parse,PROTECTED_LABELS)
    identity=selected.astype(bool)&~protected
    removed=float((selected&protected).sum()/max(1,selected.sum()))
    if not identity.any():raise ValueError('No garment remains after protected-region check')
    if removed>.15:raise ValueError(f'Identity conflicts with protected regions: {removed:.3f}')
    # Expand locally only. No arm strokes, hole filling, convex hull or all-clothes union.
    yy,xx=np.ogrid[-4:5,-4:5];kernel=xx*xx+yy*yy<=16
    inpaint=binary_dilation(identity,structure=kernel)&~protected
    return identity,protected,inpaint,removed

def save_mask(path,mask):Image.fromarray(mask.astype(np.uint8)*255).save(path)
def prepare(root,limit=None):
    bind_gpu();sys.path.insert(0,str(cutter_path()))
    from flatlay_cutter.engine import ReferenceCutter
    import torch
    c=ReferenceCutter(local_files_only=True)
    rows=manifest(root);rows=sorted(rows,key=lambda r:(r['role']!='regression',r['key']))
    if limit:rows=rows[:limit]
    for n,row in enumerate(rows):
        f=root/'data'/row['key'];done=f/'IDENTITY.json'
        if done.exists():continue
        try:
            source=read(root/'annotations/names'/(row['key']+'.json'))
            if not source['queries']:raise ValueError('No trustworthy automatic reference noun')
            sku={'key':row['key'],'references':[{'id':'product','source':str(f/'garment.png'),
                 'mask':str(f/'garment_mask.png'),'reference_queries':source['queries']}]}
            c.prepare(sku)
            im=Image.open(f/'target.png').convert('RGB');raw,diagnostics=c.cut(im,row['key'])
            parse=np.asarray(Image.open(f/'parse.png'))
            identity,protected,inpaint,removed=construct_masks(raw,parse)
            sm=np.asarray(Image.open(f/'garment_mask.png'))>127
            sp,sannotation=detail_regions(c,Image.open(f/'garment.png').convert('RGB'),sm,source['queries'])
            tp,tannotation=detail_regions(c,im,identity,source['queries'])
            smeta=sannotation['parts'];tmeta=tannotation['parts']
            for view,annotation,partmasks in [('source',sannotation,sp),('target',tannotation,tp)]:
                ap=root/'annotations'/view/(row['key']+'.json');ap.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(ap.with_suffix('.npz'),masks=partmasks)
                atomic(ap,{**annotation,'key':row['key'],'view':view,'source_sha256':digest(f/('garment.png' if view=='source' else 'target.png'))})
            save_mask(f/'identity.png',identity);save_mask(f/'protected.png',protected)
            save_mask(f/'inpaint.png',inpaint);save_mask(f/'raw_identity.png',raw)
            np.savez_compressed(f/'parts.npz',source=sp,target=tp)
            old=np.asarray(Image.open(f/'old_inpaint.png'))>127
            record={'status':'ready','queries':source['queries'],'identity_pixels':int(identity.sum()),
                    'old_inpaint_protected_pixels':int((old&protected).sum()),'new_inpaint_protected_pixels':int((inpaint&protected).sum()),
                    'old_inpaint_pixels':int(old.sum()),'new_inpaint_pixels':int(inpaint.sum()),
                    'protected_pixels':int(protected.sum()),'removed_conflict_fraction':removed,
                    'source_parts':smeta,'target_parts':tmeta,'cutter_diagnostics':diagnostics,
                    'hashes':{p:digest(f/p) for p in ['identity.png','protected.png','inpaint.png','parts.npz']}}
        except Exception as exc:record={'status':'failed','error':type(exc).__name__+': '+str(exc)}
        finally:c.refs.pop(row['key'],None)
        atomic(done,record)
        atomic(root/'MASK_PROGRESS.json',{'done':n+1,'total':len(rows),'key':row['key'],'status':record['status']})
        print('MASK',n+1,len(rows),row['key'],record['status'],record.get('error',''),flush=True)
    results={r['key']:read(root/'data'/r['key']/'IDENTITY.json')['status'] for r in rows}
    atomic(root/'MASK_SUMMARY.json',{'counts':{s:list(results.values()).count(s) for s in ['ready','failed']},'rows':results,
                                  'automatic_masks_not_ground_truth':True,'fallback_to_old_mask':False})

if __name__=='__main__':
    p=arguments(__doc__);p.add_argument('--limit',type=int);a=p.parse_args();prepare(a.root,a.limit)
