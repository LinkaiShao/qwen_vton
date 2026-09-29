"""Immutable audit splits and original inputs; previous experiment is read-only."""
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from common import *

REGRESSIONS=('test_00713_00','test_13036_00','test_11215_00','test_02023_00',
             'test_02911_00','test_04972_00','test_05329_00','test_05386_00')

def prepare(root,dataset,old):
    root.mkdir(parents=True,exist_ok=True)
    previous=read(old/'manifest.json')['records']
    used={r['source_hashes']['garment'] for r in previous}
    dev=sorted((r for r in previous if r['split']=='development'),key=lambda r:sha_order(r['key']))[:64]
    selected=[dict(r,role='calibration') for r in dev]
    for key in REGRESSIONS:
        row=next(r for r in previous if r['key']==key);selected.append(dict(row,role='regression'))
    test=[]
    for p in sorted((dataset/'test/cloth').glob('*.jpg'),key=lambda p:sha_order(p.stem)):
        h=digest(p)
        if h in used:continue
        used.add(h);iid=p.stem
        paths={'target':dataset/'test/image'/p.name,'garment':p,
               'parse':dataset/'test/image-parse-v3'/(iid+'.png'),
               'densepose':dataset/'test/image-densepose'/p.name,
               'garment_mask':dataset/'test/cloth-mask'/p.name,
               'keypoints':dataset/'test/openpose_json'/(iid+'_keypoints.json')}
        test.append({'key':'test_'+iid,'id':iid,'dataset_split':'test','role':'heldout',
                     'source_hashes':{k:digest(v) for k,v in paths.items()}})
        if len(test)==256:break
    if len(test)!=256:raise RuntimeError('Insufficient untouched garment groups')
    selected+=test
    obj={'version':2,'counts':{s:sum(r['role']==s for r in selected) for s in ['calibration','heldout','regression']},
         'dataset':str(dataset),'old_run':str(old),'parts':list(PARTS),'timesteps':list(TIMESTEPS),
         'test_rule':'Fresh official-test garment hashes; all previous train/development/test garment hashes excluded.',
         'annotation_source':'Frozen SAM3 two-prompt consensus masks. Qwen names the source garment only. Automatic proxies, not ground truth.',
         'records':selected}
    if (root/'manifest.json').exists() and read(root/'manifest.json')!=obj:raise RuntimeError('Immutable manifest changed')
    atomic(root/'manifest.json',obj)
    sys.path.insert(0,str(HERE.parent/'leffa_native'))
    from prepare import worker
    ootd=WORKSPACE/'garment_structure_eval/benchmark/vendor/ootd/run/utils_ootd.py'
    for n,row in enumerate(selected):
        folder=root/'data'/row['key']
        if not (folder/'READY.json').exists():
            prior=old/'data'/row['key']
            if prior.exists():shutil.copytree(prior,folder,dirs_exist_ok=True)
            else:worker((str(dataset),str(root),str(ootd),row))
        parse=Image.open(dataset/row['dataset_split']/'image-parse-v3'/(row['id']+'.png'))
        parse.resize(SIZE,Image.Resampling.NEAREST).save(folder/'parse.png')
        if not (folder/'old_inpaint.png').exists():shutil.copy2(folder/'mask.png',folder/'old_inpaint.png')
        if not (folder/'old_identity.png').exists():shutil.copy2(folder/'person_mask.png',folder/'old_identity.png')
    atomic(root/'DATA_READY.json',{'manifest_sha256':digest(root/'manifest.json'),'count':len(selected)})
    print('DATA_READY',len(selected),flush=True)

if __name__=='__main__':
    p=arguments(__doc__);p.add_argument('--dataset',type=Path,default=WORKSPACE/'VITON-HD-dataset')
    p.add_argument('--old',type=Path,default=DEFAULT_OLD);a=p.parse_args();prepare(a.root,a.dataset,a.old)
