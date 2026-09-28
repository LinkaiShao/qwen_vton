"""Native LeFFA attention and dense DINO supervision. No localization parameters."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

GRID = (32, 24)
LAYERS = ("up_blocks.2.attentions.2.transformer_blocks.0.attn1",
          "up_blocks.3.attentions.2.transformer_blocks.0.attn1")


def native_reference_attention(q, k, heads, grid, output_grid=GRID, chunk=128):
    """The *full-key* softmax, then the reference block; never garment-only softmax.

    Returned probabilities sum reference keys and average person queries when
    reducing resolution. Thus row sums retain absolute reference attention mass.
    Recomputed from the same projected Q/K used by fused SDPA; output AV is intact.
    """
    b, total, dim = q.shape
    h, w = grid
    n = h * w
    oh, ow = output_grid
    if total != 2*n or k.shape != q.shape or h % oh or w % ow:
        raise ValueError(f"Unsupported target/reference layout: {q.shape}, {grid}")
    q = q[:, :n].reshape(b, n, heads, dim//heads).transpose(1, 2)
    k = k.reshape(b, total, heads, dim//heads).transpose(1, 2)
    sy, sx = h//oh, w//ow

    def block(qq, kk):
        # Explicit FP32 arithmetic is essential: autocast would otherwise put
        # this matmul back in BF16 despite .float() inputs.
        with torch.autocast(device_type=qq.device.type, enabled=False):
            p = (qq.float() @ kk.float().transpose(-1, -2) / math.sqrt(dim//heads)).softmax(-1)
            ref = p[..., n:].mean(1)
            return ref.reshape(b, qq.shape[2], oh, sy, ow, sx).sum((3, 5))

    pieces = []
    for start in range(0, n, chunk):
        qq = q[:, :, start:start+chunk]
        part = checkpoint(block, qq, k, use_reentrant=False) if torch.is_grad_enabled() and (qq.requires_grad or k.requires_grad) else block(qq, k)
        pieces.append(part)
    p = torch.cat(pieces, 1).reshape(b, oh, sy, ow, sx, oh*ow).mean((2, 4))
    return p.reshape(b, oh*ow, oh*ow)


class NativeAttentionTracker:
    """Observe actual projections; optional value intervention is evaluation-only."""
    def __init__(self, unet, layers=LAYERS, image_size=(512, 384), output_grid=GRID):
        self.enabled = True
        self.layers, self.image_size, self.output_grid = tuple(layers), image_size, output_grid
        self.handles, self.captured, self.maps = [], {}, {}
        self.intervention = None  # (layer name or '*', coarse source index)
        for name in self.layers:
            module = unet.get_submodule(name)
            for kind in ("q", "k"):
                def capture(mod, inp, out, name=name, kind=kind):
                    if self.enabled:
                        self.captured[(name, kind)] = out
                self.handles.append(getattr(module, "to_"+kind).register_forward_hook(capture))

            def change_values(mod, inp, out, name=name):
                if self.intervention is None or self.intervention[0] not in (name, "*"):
                    return None
                if torch.is_grad_enabled():
                    raise RuntimeError("Causal interventions are evaluation-only")
                n = out.shape[1]//2
                h, w = self._grid(n)
                oh, ow = self.output_grid
                y, x = divmod(self.intervention[1], ow)
                value = out.clone()
                view = value[:, n:].reshape(out.shape[0], h, w, -1)
                view[:, y*h//oh:(y+1)*h//oh, x*w//ow:(x+1)*w//ow] = 0
                return value
            self.handles.append(module.to_v.register_forward_hook(change_values))

            def finish(mod, inp, out, name=name):
                if not self.enabled:
                    return
                q, k = self.captured.pop((name, "q")), self.captured.pop((name, "k"))
                self.maps[name] = native_reference_attention(q, k, mod.heads,
                    self._grid(q.shape[1]//2), self.output_grid)
            self.handles.append(module.register_forward_hook(finish))

    def _grid(self, n):
        h = round(math.sqrt(n*self.image_size[0]/self.image_size[1]))
        w = n//h
        if h*w != n:
            raise ValueError("Attention grid is not the configured rectangular image")
        return h, w

    def clear(self):
        self.captured.clear()
        self.maps.clear()

    def mean(self):
        if set(self.maps) != set(self.layers):
            raise RuntimeError("Missing native attention layer")
        return torch.stack([self.maps[name] for name in self.layers]).mean(0)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.clear()


def reliable_matches(person, garment, person_mask, garment_mask, grid=GRID):
    """Automatic clean-image pseudo-matches, not manually annotated truth.

    Retain cosine >= .5 and cycle error <= one feature cell. An empty set is
    legal: skip correspondence loss, retaining all foreground DINO supervision.
    """
    person, garment = F.normalize(person.float(), dim=-1), F.normalize(garment.float(), dim=-1)
    similarity = person @ garment.T
    similarity = similarity.masked_fill(~garment_mask.bool()[None], -1e4)
    similarity = similarity.masked_fill(~person_mask.bool()[:, None], -1e4)
    best, index = similarity.max(1)
    reverse = similarity.argmax(0)
    back = reverse[index]
    coords = torch.stack(torch.meshgrid(torch.arange(grid[0], device=person.device),
                                       torch.arange(grid[1], device=person.device), indexing="ij"), -1).reshape(-1, 2)
    cycle = (coords[back]-coords).float().norm(dim=-1)
    valid = person_mask.bool() & garment_mask.bool()[index] & (best >= .5) & (cycle <= 1)
    return {"match": index.cpu(), "reliable": valid.cpu(), "cosine": best.cpu(), "cycle": cycle.cpu()}


def correspondence_loss(attention, matches, reliable, garment_mask, grid=GRID):
    """CE on conditional native reference probabilities, with explicit masks.

    This normalization is a loss-only view; raw generation mass is also logged.
    Invalid matches never become a fake correspondence to coordinate zero.
    """
    a = attention * garment_mask[:, None].float()
    a = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
    y, x = torch.meshgrid(torch.arange(grid[0], device=a.device), torch.arange(grid[1], device=a.device), indexing="ij")
    xy = torch.stack((y, x), -1).reshape(-1, 2).float()
    target = torch.exp(-.5*(xy[None, None]-xy[matches][:, :, None]).square().sum(-1))
    target = target * garment_mask[:, None].float()
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    ce = -(target * a.clamp_min(1e-12).log()).sum(-1)
    return (ce * reliable).sum() / reliable.sum().clamp_min(1)


def dense_detail_loss(pred, target, attention, person_mask, garment_mask):
    """50% uniform coverage + 50% equal-source-patch coverage.

    The scorer compares the same target coordinates, never movable GT crops.
    Attention weights are detached here ONLY; actual generation stays connected.
    """
    error = 1-(F.normalize(pred.float(), dim=-1)*F.normalize(target.float(), dim=-1)).sum(-1)
    p = person_mask.float()
    uniform = p / p.sum(-1, keepdim=True).clamp_min(1)
    a = attention.detach() * p[:, :, None] * garment_mask[:, None].float()
    mass = a.sum(1, keepdim=True)
    visible = (mass.squeeze(1) > 1e-10) & garment_mask.bool()
    footprint = a / mass.clamp_min(1e-10)
    balanced = (footprint * visible[:, None]).sum(-1) / visible.sum(-1, keepdim=True).clamp_min(1)
    balanced = torch.where(visible.any(-1, keepdim=True), balanced, uniform)
    weights = .5*uniform + .5*balanced
    return (error*weights).sum(-1).mean(), error, weights


def routing_diagnostics(attention, garment_mask):
    """Absolute generator mass plus separately named conditional confidence."""
    foreground=attention*garment_mask[:, None].float()
    mass=foreground.sum(-1)
    conditional=foreground/mass[..., None].clamp_min(1e-12)
    entropy=-(conditional*conditional.clamp_min(1e-12).log()).sum(-1)
    normalizer=garment_mask.sum(-1,keepdim=True).clamp_min(2).float().log()
    return {'reference_mass':attention.sum(-1),'garment_foreground_mass':mass,
            'conditional_entropy':entropy/normalizer,
            'conditional_peak_probability':conditional.max(-1).values}


def reconstruct_x0(noisy, epsilon, alpha):
    return (noisy.float()-(1-alpha).sqrt()*epsilon.float()) / alpha.sqrt()


def noise_weight(alpha):
    return torch.sqrt(alpha/(1-alpha).clamp_min(1e-12)).clamp(max=1)


class DenseDINO(nn.Module):
    def __init__(self, weights, device="cuda"):
        super().__init__()
        from transformers import Dinov2Model
        self.model = Dinov2Model.from_pretrained(weights, local_files_only=True).to(device).eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485, .456, .406], device=device)[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225], device=device)[None, :, None, None])

    def forward(self, images):
        # No no_grad/detach/clamp here: predicted pixels must retain derivatives.
        rgb = F.interpolate((images.float()+1)/2, size=(448, 336), mode="bicubic", align_corners=False, antialias=True)
        out = self.model(pixel_values=(rgb-self.mean)/self.std).last_hidden_state[:, 1:]
        return F.normalize(out.float(), dim=-1)
