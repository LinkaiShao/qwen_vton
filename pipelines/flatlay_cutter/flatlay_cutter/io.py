"""Native-resolution image IO; retained RGB pixels are copied from the worn photo."""
from pathlib import Path
import hashlib, json
import numpy as np
from PIL import Image, ImageOps
try:
    import pillow_avif  # Registers AVIF support on older Pillow versions.
except ImportError:
    pass

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)

def read_image(path):
    with Image.open(path) as image:return ImageOps.exif_transpose(image).convert('RGB')

def parse_selection(raw):
    obj=None
    for i,c in enumerate(raw):
        if c!='{':continue
        try:value,_=json.JSONDecoder().raw_decode(raw[i:])
        except json.JSONDecodeError:continue
        if isinstance(value,dict) and value.get('view') in ('FLAT_LAY','PRODUCT_ONLY','WORN','DETAIL','OTHER'):obj=value
    assert obj is not None,'Missing photographic-role JSON'
    assert isinstance(obj.get('description'),str) and obj['description'].strip()
    assert isinstance(obj.get('items'),list) and len(obj['items'])<=8
    assert all(isinstance(x,str) and x.strip() and len(x)<100 for x in obj['items'])
    kind={'FLAT_LAY':'product_only','PRODUCT_ONLY':'product_only','WORN':'worn','DETAIL':'detail','OTHER':'other'}[obj['view']]
    if kind=='product_only':assert obj['items'],'Product overview needs visible item names'
    # Names in a rejected photograph cannot reach segmentation. Preserve raw evidence,
    # but let the host discard this unused field instead of turning a clear WORN/DETAIL
    # verdict into a processing error.
    else:obj=dict(obj,items=[])
    return {'kind':kind,'presentation':'flat_lay' if obj['view']=='FLAT_LAY' else 'other_product' if obj['view']=='PRODUCT_ONLY' else 'na',
            'complete':kind=='product_only','items':obj['items'],'reason':obj['description']}

def crop_box(mask,padding=12):
    assert mask.ndim==2 and mask.dtype==bool and padding>=0
    y,x=np.where(mask)
    if not len(x):raise ValueError('No product pixels: no cutout')
    h,w=mask.shape
    return [max(0,int(x.min())-padding),max(0,int(y.min())-padding),min(w,int(x.max())+1+padding),min(h,int(y.max())+1+padding)]

def export_cutout(image,mask,folder,padding=12):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rgb=np.asarray(image.convert('RGB'));assert mask.shape==rgb.shape[:2]
    box=crop_box(mask,padding);alpha=mask.astype(np.uint8)*255
    rgba=np.dstack([rgb,alpha]);rgba[~mask,:3]=0
    Image.fromarray(alpha).save(folder/'mask.png')
    Image.fromarray(rgba).crop(box).save(folder/'cutout.png')
    image.crop(box).save(folder/'crop.jpg',quality=95)
    return {'box_xyxy':box,'source_size':list(image.size),'cutout_size':[box[2]-box[0],box[3]-box[1]],
            'mask_pixels':int(mask.sum()),'alpha':'binary SAM3 segmentation; not transparency matting',
            'outputs':{name:{'file':name,'sha256':digest(folder/name)} for name in ('mask.png','cutout.png','crop.jpg')}}
