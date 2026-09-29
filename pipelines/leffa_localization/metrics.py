"""Scoring never feeds target regions back into the generator's readout."""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from functools import lru_cache

def target_support(mask,grid=(64,48),tolerance=8):
    mask=np.ascontiguousarray(mask,dtype=bool)
    return _target_support(mask.tobytes(),mask.shape,tuple(grid),tolerance)

@lru_cache(maxsize=64)
def _target_support(data,shape,grid,tolerance):
    # A target/timestep is scored against many heads. Geometry is identical;
    # cache the exact mask operation rather than repeating full-image dilation.
    mask=np.frombuffer(data,dtype=bool).reshape(shape)
    yy,xx=np.ogrid[-tolerance:tolerance+1,-tolerance:tolerance+1]
    expanded=binary_dilation(mask,structure=xx*xx+yy*yy<=tolerance*tolerance)
    # Cell-center sampling defines the 8-pixel location tolerance precisely.
    h,w=expanded.shape;ys=np.minimum(h-1,((np.arange(grid[0])+.5)*h/grid[0]).astype(int))
    xs=np.minimum(w-1,((np.arange(grid[1])+.5)*w/grid[1]).astype(int))
    support=expanded[ys[:,None],xs[None,:]];support.setflags(write=False)
    return support

def footprint(a):
    flat=np.maximum(np.asarray(a,dtype=np.float64).reshape(-1),0);total=flat.sum()
    if not np.isfinite(flat).all():raise ValueError('Nonfinite attention')
    if total<=1e-15:return {'peak_xy':None,'box_xyxy':None,'confidence':0.,'raw_mass':0.,'region':np.zeros_like(a,dtype=bool)}
    p=flat/total;order=np.argsort(-p,kind='stable');count=min(len(p),np.searchsorted(np.cumsum(p[order]),.8)+1)
    region=np.zeros_like(p,dtype=bool);region[order[:count]]=True;region=region.reshape(a.shape)
    y,x=np.unravel_index(order[0],a.shape);ys,xs=np.where(region);h,w=a.shape
    return {'peak_xy':[(x+.5)*384/w,(y+.5)*512/h],
            'box_xyxy':[int(xs.min()*384/w),int(ys.min()*512/h),int((xs.max()+1)*384/w),int((ys.max()+1)*512/h)],
            'confidence':float(p[order[:max(1,int(.05*len(p)))]].sum()),'raw_mass':float(total),
            'mean_query_mass':float(flat.mean()),'region':region}

def score_map(a,target):
    f=footprint(a);support=target_support(target,a.shape)
    if not support.any():return None
    flat=np.asarray(a,dtype=np.float64);total=flat.sum()
    y,x=np.unravel_index(np.argmax(flat),flat.shape)
    return {'peak_hit':bool(support[y,x]) if total>1e-15 else False,
            'inside_mass':float(flat[support].sum()/max(total,1e-15)),
            'confidence':f['confidence'],'raw_mass':f['raw_mass'],
            'region_iou':float((f['region']&support).sum()/max(1,(f['region']|support).sum()))}

def bootstrap(values,seed=20260929):
    values=np.asarray(values,dtype=float)
    if not len(values):return None
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(2000)])
    return {'mean':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'n':len(values)}
