"""One-time Molmo flatlay points in the trained DINO expert's normalized XY format."""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

MODEL='allenai/Molmo-7B-D-0924'
REVISION='cab33fb7f1a40091911f81165f8481920621948f'
GPU='GPU-dbb7940e-7461-387e-c0c2-a2e33f9b78ac'
CONCEPTS=['collar','lapel','cuff','button hole','button','pocket','zipper','sleeve','shoulder','logo']
PLURAL=['collars','lapels','cuffs','button holes','buttons','pockets','zippers','sleeves','shoulders','logos']
XY=re.compile(r'x(\d*)="([\d.]+)"\s+y\1="([\d.]+)"')

def parse_points(raw):
    points=[];invalid=[]
    for _,x,y in XY.findall(raw):
        x,y=float(x),float(y)
        if not (0<=x<=100 and 0<=y<=100):invalid.append([x,y]);continue
        point=[x/100,y/100]
        if point not in points:points.append(point)
    return points,invalid

def run(source,out):
    os.environ['CUDA_VISIBLE_DEVICES']=GPU;os.environ['HF_HUB_CACHE']='/mnt/nvme0/hf_home/hub'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM,AutoProcessor,GenerationConfig
    torch.set_num_threads(4);torch.backends.cuda.enable_cudnn_sdp(False)
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==GPU.removeprefix('GPU-')
    out.mkdir(parents=True,exist_ok=True);im=Image.open(source).convert('RGB');im.save(out/'flatlay.png')
    receipt={'model':MODEL,'revision':REVISION,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
             'image_size':list(im.size),'concepts':CONCEPTS,'coordinate_format':'normalized XY, same convention as StructureExperts pts',
             'segmentation_used':False,'manual_coordinate_edits':False,'gpu':GPU,'records':[]}
    print('LOADING',MODEL,REVISION,flush=True);start=time.time()
    proc=AutoProcessor.from_pretrained(MODEL,revision=REVISION,trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,trust_remote_code=True,local_files_only=True,
                    torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).cuda().eval().requires_grad_(False)
    print('LOADED',round(time.time()-start,2),flush=True)
    for concept,noun in zip(CONCEPTS,PLURAL):
        prompt=f'Point to all the {noun} on the garment.'
        inputs=proc.process(images=[im],text=prompt,images_kwargs={'max_crops':12})
        inputs={k:v.cuda().unsqueeze(0) for k,v in inputs.items()}
        if 'images' in inputs:inputs['images']=inputs['images'].to(torch.bfloat16)
        n=inputs['input_ids'].shape[1];t=time.time()
        cfg=GenerationConfig(max_new_tokens=384,do_sample=False,stop_strings='<|endoftext|>',
                             eos_token_id=proc.tokenizer.eos_token_id)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            tokens=model.generate_from_batch(inputs,cfg,tokenizer=proc.tokenizer)
        raw=proc.tokenizer.decode(tokens[0,n:],skip_special_tokens=True);points,invalid=parse_points(raw)
        rec={'concept':concept,'prompt':prompt,'points':points,'pixel_points':[[x*im.width,y*im.height] for x,y in points],
             'raw':raw,'invalid_raw_coordinates':invalid,'seconds':time.time()-t,'token_limit_reached':len(tokens[0,n:])>=384}
        receipt['records'].append(rec)
        temp=out/'points.json.tmp';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(out/'points.json')
        print('POINTS',concept,len(points),json.dumps(points),flush=True)
    receipt['complete']=True;receipt['seconds']=time.time()-start;receipt['peak_gpu_gib']=torch.cuda.max_memory_allocated()/2**30
    (out/'points.json').write_text(json.dumps(receipt,indent=2)+'\n');print('MOLMO_FLATLAY_COMPLETE',receipt['seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.source,a.out)
