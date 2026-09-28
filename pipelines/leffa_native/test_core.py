"""CPU contracts for native routing, fidelity, gradients and degenerate matches."""
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from core import (native_reference_attention, NativeAttentionTracker, reliable_matches,
                  correspondence_loss, dense_detail_loss, reconstruct_x0, noise_weight)


class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.heads=2
        self.to_q=nn.Linear(8,8,bias=False);self.to_k=nn.Linear(8,8,bias=False);self.to_v=nn.Linear(8,8,bias=False)
    def forward(self,x):
        q,k,v=[m(x).reshape(1,-1,2,4).transpose(1,2) for m in [self.to_q,self.to_k,self.to_v]]
        return F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape_as(x)


class Contracts(unittest.TestCase):
    def setUp(self):torch.manual_seed(1);torch.set_num_threads(2)

    def test_exact_probabilities_and_gradients(self):
        q=torch.randn(1,24,8,requires_grad=True);k=torch.randn_like(q,requires_grad=True)
        actual=native_reference_attention(q,k,2,(4,3),(4,3),chunk=5)
        qq=q[:,:12].reshape(1,12,2,4).transpose(1,2)
        kk=k.reshape(1,24,2,4).transpose(1,2)
        expected=(qq@kk.transpose(-1,-2)/2).softmax(-1)[...,12:].mean(1)
        torch.testing.assert_close(actual,expected)
        grads=torch.autograd.grad(actual.square().sum(),(q,k),retain_graph=True)
        ref=torch.autograd.grad(expected.square().sum(),(q,k))
        for a,b in zip(grads,ref):torch.testing.assert_close(a,b)
        self.assertTrue((actual.sum(-1)<1).all())

    def test_spatial_pooling_preserves_reference_mass(self):
        q=torch.randn(1,96,8);k=torch.randn_like(q)
        full=native_reference_attention(q,k,2,(8,6),(8,6))
        pooled=native_reference_attention(q,k,2,(8,6),(4,3))
        expected=full.reshape(1,4,2,3,2,4,2,3,2).sum((6,8)).mean((2,4)).reshape(1,12,12)
        torch.testing.assert_close(pooled,expected)

    def test_trace_parity_and_native_value_intervention(self):
        model=nn.ModuleDict({'a':Attention()});x=torch.randn(1,24,8)
        reference=model['a'](x)
        tracker=NativeAttentionTracker(model,layers=['a'],image_size=(4,3),output_grid=(4,3))
        traced=model['a'](x)
        self.assertTrue(torch.equal(traced,reference))
        before=tracker.mean().detach().clone()
        tracker.intervention=('a',3)
        with torch.no_grad():changed=model['a'](x)
        self.assertGreater(float((changed-reference).abs().max()),0)
        torch.testing.assert_close(tracker.mean(),before)
        tracker.close()

    def test_no_crop_escape_and_dino_input_gradient(self):
        p=torch.randn(1,12,8,requires_grad=True);g=torch.randn_like(p)
        a=torch.rand(1,12,12,requires_grad=True);mask=torch.ones(1,12,dtype=torch.bool)
        loss,_,weights=dense_detail_loss(p,g,a,mask,mask)
        loss.backward()
        self.assertGreater(float(p.grad.norm()),0);self.assertIsNone(a.grad)
        torch.testing.assert_close(weights.sum(-1),torch.ones(1))
        self.assertTrue((weights>=.5/12).all())

    def test_teacher_and_empty_masks(self):
        f=torch.eye(12);mask=torch.ones(12,dtype=torch.bool)
        teacher=reliable_matches(f,f,mask,mask,(4,3))
        self.assertTrue(teacher['reliable'].all())
        torch.testing.assert_close(teacher['match'],torch.arange(12))
        a=torch.rand(1,12,12,requires_grad=True)
        loss=correspondence_loss(a,teacher['match'][None],torch.zeros(1,12,dtype=torch.bool),mask[None],(4,3))
        self.assertEqual(float(loss.detach()),0);loss.backward();self.assertTrue(torch.isfinite(a.grad).all())
        teacher=reliable_matches(f,f,torch.zeros_like(mask),mask,(4,3))
        self.assertFalse(teacher['reliable'].any())

    def test_all_timestep_algebra(self):
        alpha=torch.linspace(.999,.00001,1000)[:,None,None,None]
        z=torch.randn(1000,4,2,2);eps=torch.randn_like(z)
        zt=alpha.sqrt()*z+(1-alpha).sqrt()*eps
        torch.testing.assert_close(reconstruct_x0(zt,eps,alpha),z,atol=3e-5,rtol=3e-5)
        weight=noise_weight(alpha)
        self.assertTrue(((weight>0)&(weight<=1)).all())
        self.assertTrue((weight*((1-alpha)/alpha).sqrt()<=1.00001).all())


if __name__=='__main__':unittest.main(verbosity=2)
