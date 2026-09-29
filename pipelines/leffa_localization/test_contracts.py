import unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from attention import DetailTracker,part_attention
from masks import construct_masks
from metrics import footprint,score_map
from annotate import agreement_masks,instance_masks

class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.heads=2
        self.to_q=nn.Linear(8,8,bias=False);self.to_k=nn.Linear(8,8,bias=False);self.to_v=nn.Linear(8,8,bias=False)
    def forward(self,x):
        q,k,v=[m(x).reshape(1,24,2,4).transpose(1,2) for m in (self.to_q,self.to_k,self.to_v)]
        return F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape_as(x)

class Contracts(unittest.TestCase):
    def setUp(self):torch.manual_seed(7);torch.set_num_threads(2)
    def test_exact_full_softmax_part_mass(self):
        q=torch.randn(1,24,8);k=torch.randn_like(q);parts=torch.zeros(1,2,4,3)
        parts[0,0,0,0]=1;parts[0,1,2:]=1
        got=part_attention(q,k,2,parts,output_grid=(4,3),chunk=5)
        qq=q[:,:12].reshape(1,12,2,4).transpose(1,2);kk=k.reshape(1,24,2,4).transpose(1,2)
        a=(qq@kk.transpose(-1,-2)/2).softmax(-1)[...,12:]
        want=(a@parts.flatten(2)[:,None].transpose(-1,-2)).permute(0,1,3,2).reshape_as(got)
        torch.testing.assert_close(got,want)
        self.assertTrue((got.sum(2)<1).all())
    def test_parity_and_selected_head_value_intervention(self):
        model=nn.ModuleDict({'a':Attention()});x=torch.randn(1,24,8)
        with torch.no_grad():original=model['a'](x)
        tr=DetailTracker(model,layers=['a'],output_grid=(4,3));tr.parts=torch.ones(1,2,4,3)
        with torch.no_grad():observed=model['a'](x)
        self.assertTrue(torch.equal(original,observed));before=tr.maps['a'].clone()
        mask=torch.zeros(1,1,4,3);mask[:,:,0,0]=1;tr.set_intervention('a/head1',mask,0.)
        with torch.no_grad():changed=model['a'](x)
        self.assertGreater(float((changed-original).abs().max()),0)
        torch.testing.assert_close(tr.maps['a'],before)
        torch.testing.assert_close(changed[:,:,:4],original[:,:,:4])
        tr.close()
    def test_identity_protection_holes_and_separate_pieces(self):
        raw=np.zeros((50,40),bool);raw[10:35,10:30]=True;raw[15:20,17:22]=False;raw[2:5,4:7]=True
        parse=np.zeros(raw.shape,np.uint8);parse[30:32,10:30]=14
        identity,protected,inpaint,_=construct_masks(raw,parse)
        self.assertFalse((identity&protected).any());self.assertFalse((inpaint&protected).any())
        self.assertTrue(identity[3,5]);self.assertFalse(identity[17,19])
        self.assertLess(identity.sum(),raw.sum())
    def test_conflicting_identity_rejected(self):
        raw=np.ones((20,20),bool);parse=np.zeros((20,20),np.uint8);parse[:10]=14
        with self.assertRaisesRegex(ValueError,'conflicts'):construct_masks(raw,parse)
    def test_scoring_cannot_relocate_mark_and_zero_is_not_confident(self):
        a=np.zeros((64,48));a[10,10]=1;before=footprint(a)
        target=np.zeros((512,384),bool);target[350:400,200:240]=True
        score=score_map(a,target);self.assertFalse(score['peak_hit']);self.assertEqual(score['inside_mass'],0.)
        self.assertEqual(footprint(a)['peak_xy'],before['peak_xy']);self.assertEqual(footprint(a)['peak_xy'],[84.,84.])
        self.assertEqual(footprint(np.zeros_like(a))['confidence'],0.)
    def test_teacher_disagreement_does_not_create_a_detail(self):
        a=np.zeros((30,20),bool);a[2:5,3:7]=True;b=np.roll(a,15,axis=0)
        out,matches=agreement_masks([a],[b],np.ones_like(a))
        self.assertFalse(out.any());self.assertEqual(matches,[])
        out,matches=agreement_masks([a],[a],np.ones_like(a))
        np.testing.assert_array_equal(out,a);self.assertEqual(len(matches),1)
    def test_cuff_instances_do_not_fill_torso(self):
        masks=np.zeros((4,30,20),bool);masks[2,20:25,1:4]=1;masks[2,20:25,16:19]=1
        regions,names=instance_masks(masks)
        self.assertEqual(len(names),2);self.assertFalse(regions[:,:,7:13].any())
        np.testing.assert_array_equal(regions.any(0),masks[2])

if __name__=='__main__':unittest.main(verbosity=2)
