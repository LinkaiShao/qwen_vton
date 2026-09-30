"""Disentangled structure-aware experts (paper Sec. 3.3-3.4), on cached frozen DINO features.

  e_global = P_global(f_cls)
  e_k      = P_local( T_local([q_k ; tokens from 5x5 patch windows around each grounded point of k])[0] )
All experts are L2-normalised. A pair's holistic distance weights each expert distance by a learned
softmax weight, counting only experts valid (grounded) in both images (paper Eq. 2).
Implementation details the paper leaves open are marked "choice:".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 5  # paper: fixed 5x5 regions of F_patch


class StructureExperts(nn.Module):
    def __init__(self, K: int, d_in: int = 768, grid: int = 16, d_model: int = 384, d_out: int = 256,
                 layers: int = 2, heads: int = 6, max_points: int = 4, dropout: float = 0.1):
        super().__init__()
        self.K, self.grid, self.max_points = K, grid, max_points
        self.P_global = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, d_out))
        self.w = nn.Parameter(torch.zeros(K + 1))  # holistic expert weights, softmax-normalised (Eq. 2)
        if K == 0:
            return
        self.inp = nn.Linear(d_in, d_model)
        # choice: learnable window-offset and point-index embeddings so the encoder knows token layout
        self.offset = nn.Parameter(torch.randn(WIN * WIN, d_model) * 0.02)
        self.pidx = nn.Parameter(torch.randn(max_points, d_model) * 0.02)
        self.q = nn.Parameter(torch.randn(K, d_model) * 0.02)  # learnable concept-specific queries q_k
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.T_local = nn.TransformerEncoder(layer, layers)  # shared across concepts (paper)
        self.P_local = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_out))

    def forward(self, cls, patch, pts, pmask):
        """cls (B,D); patch (B,G*G,D); pts (B,K,P,2) in 0..1; pmask (B,K,P) True where a point exists.
        -> e_global (B,d_out), e_local (B,K,d_out), valid (B,K)"""
        B = cls.shape[0]
        e_global = F.normalize(self.P_global(cls.float()), dim=-1)
        if self.K == 0:
            return e_global, cls.new_zeros(B, 0, e_global.shape[1]), torch.zeros(B, 0, dtype=torch.bool, device=cls.device)
        valid = pmask.any(-1)
        e_local = e_global.new_zeros(B, self.K, e_global.shape[1])
        bk = valid.nonzero()
        if bk.numel():
            b, k = bk[:, 0], bk[:, 1]
            G, P = self.grid, self.max_points
            xy, pm = pts[b, k], pmask[b, k]                                   # (Nv,P,2), (Nv,P)
            gx = (xy[..., 0] * G).long().clamp(0, G - 1)
            gy = (xy[..., 1] * G).long().clamp(0, G - 1)
            off = torch.arange(-(WIN // 2), WIN // 2 + 1, device=cls.device)
            wy = (gy[..., None, None] + off[:, None]).clamp(0, G - 1)          # (Nv,P,5,1)
            wx = (gx[..., None, None] + off[None, :]).clamp(0, G - 1)          # (Nv,P,1,5)
            flat = (wy * G + wx).reshape(len(b), P, WIN * WIN)                # (Nv,P,25)
            feats = patch[b[:, None, None], flat].float()                      # (Nv,P,25,D)
            tok = self.inp(feats) + self.offset + self.pidx[:, None, :]
            tok = tok.reshape(len(b), P * WIN * WIN, -1)
            pad = (~pm)[..., None].expand(len(b), P, WIN * WIN).reshape(len(b), P * WIN * WIN)
            x = torch.cat([self.q[k][:, None, :], tok], 1)
            kpm = torch.cat([torch.zeros(len(b), 1, dtype=torch.bool, device=cls.device), pad], 1)
            h = self.T_local(x, src_key_padding_mask=kpm)[:, 0]
            e_local[b, k] = F.normalize(self.P_local(h), dim=-1)
        return e_global, e_local, valid

    def expert_distances(self, ga, la, va, gb, lb, vb):
        """Pairwise distances between set a (Na) and set b (Nb).
        -> D (Na,Nb,K+1) with global first; V (Na,Nb,K+1) validity (global always valid)."""
        dg = torch.cdist(ga, gb)[..., None]
        if self.K == 0:
            return dg, torch.ones_like(dg, dtype=torch.bool)
        # floor inside the sqrt: a pair at distance exactly 0 (an image with itself, or a duplicated view) has an
        # infinite sqrt derivative, and 0 * inf = NaN in backward poisons every weight even though the loss ignores it
        dl = torch.sqrt(torch.clamp(2 - 2 * torch.einsum("akd,bkd->abk", la, lb), min=1e-6))
        vl = va[:, None, :] & vb[None, :, :]
        D = torch.cat([dg, dl], -1)
        V = torch.cat([torch.ones_like(dg, dtype=torch.bool), vl], -1)
        return D, V

    def holistic(self, D, V):
        w = torch.softmax(self.w, 0)
        wv = w * V.float()
        return (wv * D).sum(-1) / wv.sum(-1).clamp_min(1e-8)
