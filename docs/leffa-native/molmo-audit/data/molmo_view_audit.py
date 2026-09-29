"""Paired, fixed-prompt Molmo audit: raw predictions and blinded review sheets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont
from molmo_flatlay import MODEL, REVISION, GPU, CONCEPTS, PLURAL, parse_points


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def sheets(root):
    manifest=read(root/'manifest.json');out=root/'blind';out.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    for i,row in enumerate(manifest['records']):
        sheet=Image.new('RGB',(768*3,1084),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for j,(view,path) in enumerate(row['views'].items()):
            im=Image.open(path).convert('RGB').resize((768,1024))
            sheet.paste(im,(j*768,60));draw.text((j*768+15,15),f'{i+1:02d} {row["id"]} · {view}',font=font,fill='black')
        sheet.save(out/f'{i+1:02d}_{row["id"]}.jpg',quality=94)
    print('BLIND_SHEETS',len(manifest['records']),flush=True)


def infer(root):
    os.environ['CUDA_VISIBLE_DEVICES']=GPU
    os.environ['HF_HUB_CACHE']='/mnt/nvme0/hf_home/hub'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from transformers import AutoModelForCausalLM,AutoProcessor,GenerationConfig
    torch.set_num_threads(4);torch.backends.cuda.enable_cudnn_sdp(False)
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==GPU.removeprefix('GPU-')
    manifest=read(root/'manifest.json');manifest_sha=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    start=time.time();print('LOADING',MODEL,flush=True)
    proc=AutoProcessor.from_pretrained(MODEL,revision=REVISION,trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,trust_remote_code=True,local_files_only=True,
          torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).cuda().eval().requires_grad_(False)
    cfg=GenerationConfig(max_new_tokens=384,do_sample=False,stop_strings='<|endoftext|>',eos_token_id=proc.tokenizer.eos_token_id)
    jobs=[(row,view,path) for row in manifest['records'] for view,path in row['views'].items()]
    for num,(row,view,path) in enumerate(jobs,1):
        dest=root/'predictions'/f'{row["id"]}__{view}.json'
        if dest.exists() and read(dest).get('complete'):
            old=read(dest)
            if old['manifest_sha256']!=manifest_sha:raise ValueError('Manifest changed')
            continue
        source=Path(path)
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['hashes'][view]:raise ValueError('Image changed')
        im=Image.open(source).convert('RGB');beg=time.time()
        records=[]
        for concept,noun in zip(CONCEPTS,PLURAL):
            prompt=f'Point to all the {noun} on the garment.'
            inputs=proc.process(images=[im],text=prompt,images_kwargs={'max_crops':12})
            inputs={k:v.cuda().unsqueeze(0) for k,v in inputs.items()}
            if 'images' in inputs:inputs['images']=inputs['images'].to(torch.bfloat16)
            n=inputs['input_ids'].shape[1]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                tokens=model.generate_from_batch(inputs,cfg,tokenizer=proc.tokenizer)
            raw=proc.tokenizer.decode(tokens[0,n:],skip_special_tokens=True)
            points,invalid=parse_points(raw)
            records.append({'concept':concept,'prompt':prompt,'points':points,'raw':raw,
                'invalid_raw_coordinates':invalid,'token_limit_reached':len(tokens[0,n:])>=384,
                'unparsed_point_tag':'<point' in raw and not points})
        atomic(dest,{'complete':True,'id':row['id'],'view':view,'image_size':list(im.size),
            'model':MODEL,'revision':REVISION,'source':str(source),'image_sha256':row['hashes'][view],
            'manifest_sha256':manifest_sha,'records':records,'seconds':time.time()-beg})
        progress={'stage':'inference','done':num,'total':len(jobs),'seconds':time.time()-start,
                  'last_id':row['id'],'last_view':view,'gpu':GPU,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
        atomic(root/'PROGRESS.json',progress);print('IMAGE_DONE',num,len(jobs),row['id'],view,round(time.time()-beg,2),flush=True)
    atomic(root/'INFERENCE_COMPLETE.json',{'count':len(jobs),'seconds':time.time()-start,'manifest_sha256':manifest_sha,
          'gpu':GPU,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
    print('INFERENCE_COMPLETE',len(jobs),time.time()-start,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['sheets','infer'])
    p.add_argument('--root',type=Path,default=Path('/mnt/nvme0/leffa_native/molmo_view_audit_20260929'))
    a=p.parse_args();(sheets if a.stage=='sheets' else infer)(a.root)
