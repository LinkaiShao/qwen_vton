"""Prepare all VITON-HD pairs without Molmo/part boxes; deterministic hash splits."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def worker(args):
    dataset, output, ootd, row = args
    source = Path(dataset)/row['dataset_split']
    dest = Path(output)/'data'/row['key']
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'READY.json').exists():
        existing = json.loads((dest/'READY.json').read_text())
        if existing.get('source_hashes') == row['source_hashes']:
            return row['key']
        raise ValueError('Prepared source changed: '+row['key'])
    iid = row['id']
    for name, subdir in [('target','image'),('garment','cloth'),('densepose','image-densepose')]:
        Image.open(source/subdir/(iid+'.jpg')).convert('RGB').resize((384,512),Image.Resampling.LANCZOS).save(dest/(name+'.png'))
    parse = Image.open(source/'image-parse-v3'/(iid+'.png'))
    labels = np.asarray(parse)
    if labels.ndim != 2 or labels.max() >= 20:
        raise ValueError('Expected LIP parsing: '+iid)
    garment_mask = Image.open(source/'cloth-mask'/(iid+'.jpg')).convert('L')
    garment_mask.point(lambda x: 255 if x>=128 else 0).resize((384,512),Image.Resampling.NEAREST).save(dest/'garment_mask.png')
    # Visible upper-body garment, NOT the larger inpainting mask.
    pmask = Image.fromarray((np.isin(labels,[5,6,7])*255).astype(np.uint8))
    pmask.resize((384,512),Image.Resampling.NEAREST).save(dest/'person_mask.png')
    global _ootd
    if '_ootd' not in globals():
        spec = importlib.util.spec_from_file_location('native_ootd_recipe',ootd)
        _ootd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ootd)
    lut=np.array([0,1,2,0,3,4,7,4,0,6,7,17,5,11,14,15,12,13,9,10],dtype=np.uint8)
    people=json.loads((source/'openpose_json'/(iid+'_keypoints.json')).read_text())['people']
    pose=np.asarray(people[0]['pose_keypoints_2d']).reshape(-1,3)
    xy=pose[:,:2].copy()*[384/parse.width,512/parse.height]
    xy[pose[:,2]<=0]=0
    mask,_=_ootd.get_mask_location('hd','upper_body',Image.fromarray(lut[labels]),{'pose_keypoints_2d':xy.reshape(-1).tolist()})
    mask.resize((384,512),Image.Resampling.NEAREST).save(dest/'mask.png')
    atomic(dest/'READY.json',dict(row, visible_garment_pixels=int(np.count_nonzero(np.asarray(pmask)))))
    return row['key']


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for split in ['train','test']:
        for path in sorted((args.dataset/split/'image').glob('*.jpg')):
            iid=path.stem
            paths={'target':path,'garment':args.dataset/split/'cloth'/(iid+'.jpg'),
                   'parse':args.dataset/split/'image-parse-v3'/(iid+'.png'),
                   'densepose':args.dataset/split/'image-densepose'/(iid+'.jpg'),
                   'garment_mask':args.dataset/split/'cloth-mask'/(iid+'.jpg'),
                   'keypoints':args.dataset/split/'openpose_json'/(iid+'_keypoints.json')}
            records.append(dict(id=iid,key=split+'_'+iid,dataset_split=split,
                                source_hashes={k:digest(v) for k,v in paths.items()}))
    train=[r for r in records if r['dataset_split']=='train']
    groups={}
    for r in train:groups.setdefault(r['source_hashes']['garment'],[]).append(r)
    order=sorted(groups,key=lambda h:hashlib.sha256(('native-v1-dev:'+h).encode()).hexdigest())
    dev=set();count=0
    for h in order:
        if count>=512:break
        dev.add(h);count+=len(groups[h])
    for row in train:row['split']='development' if row['source_hashes']['garment'] in dev else 'train'
    tests=[r for r in records if r['dataset_split']=='test' and r['source_hashes']['garment'] not in groups]
    tests.sort(key=lambda r:hashlib.sha256(('native-v1-test:'+r['id']).encode()).hexdigest())
    selected=[];seen=set()
    for row in tests:
        if row['source_hashes']['garment'] in seen:continue
        row['split']='test';selected.append(row);seen.add(row['source_hashes']['garment'])
        if len(selected)==256:break
    if len(selected)!=256:raise RuntimeError('Insufficient garment-separated official test examples')
    rows=train+selected
    manifest={'version':1,'image_size':[512,384],'dino_grid':[32,24],
              'labels':'No named parts or Molmo boxes; existing garment parsing and automatic DINO correspondences.',
              'split_rule':'Whole byte-identical garment hash groups; deterministic SHA256 order; no test/train identical garment hashes.',
              'counts':{s:sum(r['split']==s for r in rows) for s in ['train','development','test']},'records':rows}
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Immutable manifest changed')
    atomic(path,manifest)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=((str(args.dataset),str(args.output),str(args.ootd_utils),row) for row in rows)
        for i,key in enumerate(pool.map(worker,jobs,chunksize=8)):
            if i%100==0:print('PREPARED',i+1,len(rows),key,flush=True)
    atomic(args.output/'DATA_READY.json',{'manifest_sha256':digest(path),'counts':manifest['counts']})
    print('DATA_READY',manifest['counts'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['dataset','output','ootd-utils']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    prepare(p.parse_args())
