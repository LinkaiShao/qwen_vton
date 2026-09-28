"""Frozen SAM3 loader and the reviewed colour prefilter."""
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

SAM_MODEL = 'facebook/sam3'
SAM_REVISION = '3c879f39826c281e95690f02c7821c4de09afae7'

def lab_pixels(image,mask):
    import cv2
    rgb=np.asarray(image.resize((160,160)),dtype=np.float32)/255
    keep=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((160,160),Image.Resampling.NEAREST))>0
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)[keep]

def reference_palette(images):
    import cv2
    px=np.concatenate([lab_pixels(im,mask) for im,mask in images])
    assert len(px)>0,'Empty reference mask'
    cv2.setRNGSeed(42)
    _,labels,centers=cv2.kmeans(px.astype(np.float32),min(16,len(px)),None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,40,.1),1,cv2.KMEANS_PP_CENTERS)
    weights=np.bincount(labels[:,0],minlength=len(centers))/len(px)
    # Tiny tags/background contamination must not authorize a different colorway.
    centers=centers[weights>=.02];centers[:,0]*=.5
    return centers

def color_support(image,mask,palette):
    px=lab_pixels(image,mask)
    if not len(px):return 0.
    px[:,0]*=.5
    dist=cKDTree(palette).query(px,workers=1)[0]
    return float(np.mean(dist<=22.))


class DirectCutter:
    def __init__(self, model=SAM_MODEL, revision=SAM_REVISION, local_files_only=False):
        import torch
        from transformers import Sam3Model, Sam3Processor
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('A CUDA GPU with BF16 support is required. Select it with CUDA_VISIBLE_DEVICES.')
        torch.set_num_threads(4)
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.proc = Sam3Processor.from_pretrained(model, revision=revision, local_files_only=local_files_only)
        self.net = Sam3Model.from_pretrained(model, revision=revision, local_files_only=local_files_only, dtype=torch.bfloat16).cuda().eval()
        self.refs = {}
