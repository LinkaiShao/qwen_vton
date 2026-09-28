"""Real LeFFA single-timestep component localization and differentiable DINO scoring.

LeFFA disables attn2. We attach semantic component keys to its live attn1 query
features as a readout; we do not pretend that pretrained LeFFA has text tokens.
The denoiser's output is unchanged by installing the hooks. Training changes
selected existing image-attention Q/K projections and the new semantic keys.
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

CONCEPTS = ("collar_neckline", "left_sleeve_cuff", "right_sleeve_cuff")
LAYERS = ("up_blocks.2.attentions.2.transformer_blocks.0.attn1",
          "up_blocks.3.attentions.2.transformer_blocks.0.attn1")


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_boxes(boxes):
    if not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any():
        raise ValueError("Boxes must be finite normalized xyxy coordinates")
    if (boxes[..., 2:] <= boxes[..., :2]).any():
        raise ValueError("Boxes must have positive width and height")


def box_coverage(boxes, grid_size):
    """Fraction of each cell covered by each box; supports rectangular grids."""
    validate_boxes(boxes)
    height, width = grid_size
    x0 = torch.arange(width, device=boxes.device).float() / width
    y0 = torch.arange(height, device=boxes.device).float() / height
    dx = (torch.minimum(x0 + 1 / width, boxes[..., 2, None]) -
          torch.maximum(x0, boxes[..., 0, None])).clamp_min(0) * width
    dy = (torch.minimum(y0 + 1 / height, boxes[..., 3, None]) -
          torch.maximum(y0, boxes[..., 1, None])).clamp_min(0) * height
    return dy[..., :, None] * dx[..., None, :]


def spatial_mass(heatmaps, boxes):
    mask = box_coverage(boxes, heatmaps.shape[-2:])
    return (heatmaps * mask).sum((-2, -1)) / heatmaps.sum((-2, -1)).clamp_min(1e-8)


def compute_spatial_mass_loss(attn_map, bboxes, grid_size=None, valid=None):
    if attn_map.ndim == 2:
        if grid_size is None:
            raise ValueError("Flat attention needs the actual rectangular grid_size")
        attn_map = attn_map.reshape(attn_map.shape[0], *grid_size)
    mass = spatial_mass(attn_map, bboxes)
    return (1 - mass[valid]).mean() if valid is not None else (1 - mass).mean()


def crop_regions(images, boxes, size=224):
    """Differentiable crop/resize: gradients reach image pixels AND box locations."""
    validate_boxes(boxes)
    batch, count, _ = boxes.shape
    if batch != images.shape[0]:
        raise ValueError("Image/box batch mismatch")
    uv = (torch.arange(size, device=images.device, dtype=torch.float32) + .5) / size
    x = boxes[..., 0, None] + uv * (boxes[..., 2] - boxes[..., 0])[..., None]
    y = boxes[..., 1, None] + uv * (boxes[..., 3] - boxes[..., 1])[..., None]
    grid = torch.stack((x[..., None, :].expand(-1, -1, size, -1),
                        y[..., :, None].expand(-1, -1, -1, size)), -1) * 2 - 1
    expanded = images[:, None].expand(-1, count, -1, -1, -1).reshape(batch * count, *images.shape[1:])
    return F.grid_sample(expanded.float(), grid.reshape(batch * count, size, size, 2),
                         mode="bilinear", padding_mode="border", align_corners=False)


def attention_boxes(heatmaps, sizes):
    """Soft attention center; fixed component window sizes from training metadata."""
    height, width = heatmaps.shape[-2:]
    x = (torch.arange(width, device=heatmaps.device).float() + .5) / width
    y = (torch.arange(height, device=heatmaps.device).float() + .5) / height
    p = heatmaps / heatmaps.sum((-2, -1), keepdim=True).clamp_min(1e-8)
    cx = (p.sum(-2) * x).sum(-1)
    cy = (p.sum(-1) * y).sum(-1)
    half = sizes / 2
    centers = torch.stack((cx, cy), -1)
    centers = torch.maximum(half, torch.minimum(1 - half, centers))
    return torch.cat((centers - half, centers + half), -1)


def reconstruct_x0(noisy, prediction, alphas_cumprod, timestep, prediction_type):
    alpha = alphas_cumprod[timestep].float().reshape(-1, 1, 1, 1)
    if prediction_type == "epsilon":
        return (noisy.float() - (1 - alpha).sqrt() * prediction.float()) / alpha.sqrt().clamp_min(1e-8)
    if prediction_type == "v_prediction":
        return alpha.sqrt() * noisy.float() - (1 - alpha).sqrt() * prediction.float()
    if prediction_type == "sample":
        return prediction.float()
    raise ValueError("Unsupported scheduler prediction type: " + prediction_type)


class AttentionTracker(nn.Module):
    """Semantic-key attention readout using live image-attention Q projections.

    Forward hooks observe Q tensors, not attn2 outputs. Maps remain attached to
    autograd. No annotation boxes are passed to hooks, logits, or model forward.
    Normalization is over target spatial positions for each component key.
    """
    def __init__(self, unet, layer_names=LAYERS):
        super().__init__()
        self.modules_by_name = {n: unet.get_submodule(n) for n in layer_names}
        self.component_keys = nn.ParameterDict()
        self.captured = {}
        self.handles = []
        self.enabled = True
        for index, (name, attn) in enumerate(self.modules_by_name.items()):
            key = str(index)
            self.component_keys[key] = nn.Parameter(torch.randn(len(CONCEPTS), attn.to_k.in_features) * .02)
            def observe(module, inputs, output, layer=name):
                if self.enabled:
                    self.captured[layer] = output
            self.handles.append(attn.to_q.register_forward_hook(observe))

    def clear(self):
        self.captured.clear()

    def maps(self, image_size):
        result = []
        for index, (name, attn) in enumerate(self.modules_by_name.items()):
            q = self.captured[name]
            batch, tokens, channels = q.shape
            if tokens % 2:
                raise ValueError("Expected concatenated target and garment spatial tokens")
            count = tokens // 2
            height = round(math.sqrt(count * image_size[0] / image_size[1]))
            width = count // height
            if height * width != count:
                raise ValueError("Attention grid does not match image aspect ratio")
            dim = channels // attn.heads
            q = q[:, :count].reshape(batch, count, attn.heads, dim).permute(0, 2, 1, 3)
            semantic_k = attn.to_k(self.component_keys[str(index)]).reshape(len(CONCEPTS), attn.heads, dim)
            logits = torch.einsum("bhnd,khd->bhkn", q.float(), semantic_k.float()) / math.sqrt(dim)
            weights = logits.softmax(-1).mean(1).reshape(batch, len(CONCEPTS), height, width)
            result.append(weights)
        # Rescale each map to the finest grid and renormalize probability mass.
        shape = result[-1].shape[-2:]
        result = [F.interpolate(x, shape, mode="bilinear", align_corners=False) if x.shape[-2:] != shape else x
                  for x in result]
        result = [x / x.sum((-2, -1), keepdim=True).clamp_min(1e-8) for x in result]
        return torch.stack(result).mean(0)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()


class DINOComponentEvaluator(nn.Module):
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        from transformers import Dinov2Model
        self.dino = Dinov2Model.from_pretrained(model_path, local_files_only=True).to(device).eval()
        self.dino.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485, .456, .406], device=device)[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225], device=device)[None, :, None, None])

    def features(self, crops):
        # Frozen parameters still permit input gradients; no no_grad decorator here.
        crops = ((crops.float() + 1) / 2).clamp(0, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = self.dino(pixel_values=(crops - self.mean) / self.std).last_hidden_state[:, 0]
        return F.normalize(features.float(), dim=-1)

    def extract_crop_features(self, images, boxes):
        return self.features(crop_regions(images, boxes))

    def distances(self, images, boxes, target_features):
        predicted = self.extract_crop_features(images, boxes)
        return 1 - (predicted * target_features).sum(-1)


def image_tensor(path, size=None, mask=False):
    image = Image.open(path).convert("L" if mask else "RGB")
    if size is not None:
        image = image.resize(size[::-1], Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
    array = np.asarray(image).copy()
    value = torch.from_numpy(array).float()
    if mask:
        return (value[None, None] > 127).float()
    return value.permute(2, 0, 1)[None] / 127.5 - 1


def save_image(tensor, path):
    image = ((tensor[0].detach().float().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def load_models(args):
    sys.path.insert(0, str(args.leffa_code))
    from leffa.model import LeffaModel
    from diffusers import DDPMScheduler
    root = args.leffa_weights
    started = time.time()
    print("LOAD_LEFFA", flush=True)
    with torch.device("meta"):
        model = LeffaModel(str(root / "stable-diffusion-inpainting"), pretrained_model="",
                           dtype="float32", height=args.height, width=args.width)
    model.noise_scheduler = DDPMScheduler.from_pretrained(str(root / "stable-diffusion-inpainting"), subfolder="scheduler")
    state = torch.load(root / "virtual_tryon.pth", map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, assign=True)
    del state
    model.to("cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    projections = []
    for name in LAYERS:
        attention = model.unet.get_submodule(name)
        for field in ("to_q", "to_k"):
            projection = getattr(attention, field).float().requires_grad_(True)
            projections.extend(projection.parameters())
    tracker = AttentionTracker(model.unet).cuda()
    evaluator = DINOComponentEvaluator(str(args.dino_weights))
    print("MODELS_READY", round(time.time() - started, 1), "seconds", flush=True)
    return model, tracker, evaluator, projections


def cache_inputs(args, model, evaluator, manifest):
    cache = []
    size = (args.height, args.width)
    for row in manifest["records"]:
        folder = args.data / "data" / row["id"]
        images = {name: image_tensor(folder / (name + ".png"), size).cuda()
                  for name in ("target", "garment", "densepose")}
        mask = image_tensor(folder / "mask.png", size, mask=True).cuda()
        boxes = torch.tensor(row["boxes"], device="cuda", dtype=torch.float32)[None]
        valid = torch.tensor(row["visible"], device="cuda", dtype=torch.bool)[None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            def encode(image):
                return model.vae.encode(image.to(torch.bfloat16)).latent_dist.mode() * model.vae.config.scaling_factor
            z0 = encode(images["target"])
            garment = encode(images["garment"])
            masked = encode(images["target"] * (mask < .5))
            target_features = evaluator.extract_crop_features(images["target"], boxes)
        cache.append(dict(row=row, boxes=boxes, valid=valid, target=images["target"], z0=z0,
                          garment=garment, masked=masked,
                          mask=F.interpolate(mask, z0.shape[-2:], mode="nearest"),
                          pose=F.interpolate(images["densepose"], z0.shape[-2:], mode="nearest"),
                          target_features=target_features.detach()))
    print("INPUTS_CACHED", len(cache), flush=True)
    return cache


class Experiment:
    def __init__(self, args, manifest):
        self.args, self.manifest = args, manifest
        self.model, self.tracker, self.evaluator, self.projections = load_models(args)
        self.cache = cache_inputs(args, self.model, self.evaluator, manifest)
        self.train_rows = [x for x in self.cache if x["row"]["split"] == "train"]
        self.val_rows = [x for x in self.cache if x["row"]["split"] == "validation"]
        self.alpha = self.model.noise_scheduler.alphas_cumprod.cuda()
        self.prediction_type = self.model.noise_scheduler.config.prediction_type
        if self.prediction_type != "epsilon":
            raise ValueError("This LeFFA checkpoint was expected to use epsilon prediction")
        self.steps = len(self.alpha)
        self.calls = {"denoiser": 0, "reference_encoder": 0}
        self.model.unet.register_forward_hook(lambda *a: self.count("denoiser"))
        self.model.unet_encoder.register_forward_hook(lambda *a: self.count("reference_encoder"))
        self.initial_proj = [p.detach().cpu().clone() for p in self.projections]
        self.initial_keys = {k: v.detach().cpu().clone() for k, v in self.tracker.state_dict().items()}
        self.initial_dino = self.evaluator.dino.embeddings.cls_token.detach().clone()
        self.oom_count = 0

    def count(self, name):
        self.calls[name] += 1

    def reset(self):
        with torch.no_grad():
            for p, original in zip(self.projections, self.initial_proj):
                p.copy_(original.to(p.device))
        self.tracker.load_state_dict(self.initial_keys)
        self.tracker.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def predict(self, row, timestep, seed):
        self.tracker.clear()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(row["z0"].shape, device="cuda", generator=generator, dtype=torch.float32)
        t = torch.tensor([timestep], device="cuda", dtype=torch.long)
        alpha = self.alpha[t].reshape(-1, 1, 1, 1)
        noisy = alpha.sqrt() * row["z0"].float() + (1 - alpha).sqrt() * noise
        before = dict(self.calls)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, features = self.model.unet_encoder(row["garment"], t, encoder_hidden_states=None, return_dict=False)
        inputs = torch.cat((noisy, row["mask"], row["masked"], row["pose"]), 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.model.unet(inputs, t, encoder_hidden_states=None,
                                   reference_features=list(features), return_dict=False)[0]
            maps = self.tracker.maps((self.args.height, self.args.width)) if self.tracker.enabled else None
        if self.calls["denoiser"] - before["denoiser"] != 1 or self.calls["reference_encoder"] - before["reference_encoder"] != 1:
            raise AssertionError("Expected one reference encoder and one denoiser call, without sampling loop")
        x0 = reconstruct_x0(noisy, pred, self.alpha, t, self.prediction_type)
        return pred, noise, x0, maps

    def decode(self, x0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.vae.decode(x0.to(torch.bfloat16) / self.model.vae.config.scaling_factor).sample.float()

    def detail_loss(self, row, image, maps):
        # Half of the loss evaluates the exact labelled site. Half follows the
        # predicted attention location, keeping a spatially anchored comparison.
        sizes = row["boxes"][..., 2:] - row["boxes"][..., :2]
        predicted_boxes = attention_boxes(maps, sizes)
        fixed = self.evaluator.distances(image, row["boxes"], row["target_features"])
        pointed = self.evaluator.distances(image, predicted_boxes, row["target_features"])
        valid = row["valid"].flatten()
        return .5 * (fixed[valid].mean() + pointed[valid].mean()), fixed, pointed, predicted_boxes

    def verify(self):
        row = self.train_rows[0]
        with torch.no_grad():
            self.tracker.enabled = False
            a = self.predict(row, 500, 110)[0]
            self.tracker.enabled = True
            b = self.predict(row, 500, 110)[0]
            delta = float((a - b).abs().max())
        if delta != 0:
            raise AssertionError("Hooks changed denoiser output")
        pred, noise, x0, maps = self.predict(row, 500, 111)
        image = self.decode(x0)
        loss, fixed, pointed, _ = self.detail_loss(row, image, maps)
        grads = torch.autograd.grad(loss, self.projections + list(self.tracker.parameters()), allow_unused=True)
        projection_norm = sum(float(g.float().square().sum()) for g in grads[:len(self.projections)] if g is not None) ** .5
        key_norm = sum(float(g.float().square().sum()) for g in grads[len(self.projections):] if g is not None) ** .5
        if not math.isfinite(projection_norm) or projection_norm <= 0 or key_norm <= 0:
            raise AssertionError("DINO feedback does not reach projections and localization keys")
        result = {"hook_output_max_abs_difference": delta, "prediction_type": self.prediction_type,
                  "dino_gradient_norm_existing_qk_projections": projection_norm,
                  "dino_gradient_norm_component_keys": key_norm,
                  "dino_parameters_frozen": all(not p.requires_grad for p in self.evaluator.parameters()),
                  "vae_parameters_frozen": all(not p.requires_grad for p in self.model.vae.parameters()),
                  "denoiser_calls_per_prediction": 1, "reference_encoder_calls_per_prediction": 1,
                  "image_size": list(image.shape), "grid_size": list(maps.shape[-2:]),
                  "dino_distance": float(loss.detach()), "passed": True}
        atomic(self.args.output / "GRADIENT_VERIFICATION.json", result)
        print("GRADIENT_VERIFIED", json.dumps(result), flush=True)
        del pred, noise, x0, maps, image, loss, grads
        self.tracker.clear()
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(self, tag, timesteps=(.25, .5, .75), save=False):
        records = []
        all_maps, all_boxes, all_valid = [], [], []
        for row in self.val_rows:
            for frac in timesteps:
                t = min(self.steps - 1, round(frac * (self.steps - 1)))
                seed = int(hashlib.sha256((row["row"]["id"] + str(t)).encode()).hexdigest()[:8], 16)
                pred, noise, x0, maps = self.predict(row, t, seed)
                image = self.decode(x0)
                loss, fixed, pointed, boxes = self.detail_loss(row, image, maps)
                mass = spatial_mass(maps, row["boxes"])
                if frac == .5:
                    all_maps.append(maps.cpu());all_boxes.append(row["boxes"].cpu());all_valid.append(row["valid"].cpu())
                for index, concept in enumerate(CONCEPTS):
                    if not row["valid"][0, index]:
                        continue
                    records.append({"id": row["row"]["id"], "concept": concept, "t_fraction": frac,
                                    "timestep": t, "mass_inside_box": float(mass[0, index]),
                                    "fixed_roi_dino_distance": float(fixed[index]),
                                    "pointed_roi_dino_distance": float(pointed[index]),
                                    "box": row["boxes"][0, index].tolist(),
                                    "predicted_box": boxes[0, index].tolist()})
                if save and frac == .5:
                    folder = self.args.output / "examples" / tag / row["row"]["id"]
                    save_image(image, folder / "single_step_prediction.png")
                    save_image(row["target"], folder / "target.png")
                    np.savez_compressed(folder / "attention.npz", maps=maps.cpu().numpy(),
                                        boxes=row["boxes"].cpu().numpy(), valid=row["valid"].cpu().numpy())
        def average(key, items):
            return sum(x[key] for x in items) / len(items) if items else None
        at_half = [r for r in records if r["t_fraction"] == .5]
        summary = {"tag": tag, "records": records,
                   "mean_fixed_roi_dino_distance": average("fixed_roi_dino_distance", records),
                   "mean_pointed_roi_dino_distance": average("pointed_roi_dino_distance", records),
                   "mass_at_half": {c: average("mass_inside_box", [r for r in at_half if r["concept"] == c]) for c in CONCEPTS},
                   "fraction_regions_over_80pct_mass_at_half": sum(r["mass_inside_box"] > .8 for r in at_half) / max(1, len(at_half))}
        if all_maps:
            maps, boxes, valid = torch.cat(all_maps), torch.cat(all_boxes), torch.cat(all_valid)
            shuffled = spatial_mass(maps.roll(1, 0), boxes)
            summary["cross_image_shuffled_map_mass_at_half"] = float(shuffled[valid].mean())
            summary["own_image_mass_at_half"] = float(spatial_mass(maps, boxes)[valid].mean())
        atomic(self.args.output / "evaluations" / (tag + ".json"), summary)
        print("EVAL", tag, json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)
        self.tracker.clear()
        return summary

    def train_arm(self, name, use_dino):
        self.reset()
        optimizer = torch.optim.AdamW([
            {"params": self.projections, "lr": self.args.lr},
            {"params": self.tracker.parameters(), "lr": self.args.query_lr}], weight_decay=.01)
        rng = random.Random(self.args.seed)
        schedule = []
        while len(schedule) < self.args.steps:
            block = list(range(len(self.train_rows)));rng.shuffle(block);schedule.extend(block)
        start = time.time()
        log = self.args.output / (name + "_train.jsonl")
        for step in range(self.args.steps):
            row = self.train_rows[schedule[step]]
            noise_seed = self.args.seed + step * 7919
            t = random.Random(noise_seed).randrange(self.steps)
            optimizer.zero_grad(set_to_none=True)
            pred, noise, x0, maps = self.predict(row, t, noise_seed)
            denoise = F.mse_loss(pred.float(), noise)
            localization = compute_spatial_mass_loss(maps, row["boxes"], valid=row["valid"])
            # The control also runs cropping/DINO on every iteration, but does
            # not backpropagate its score. No cached/mock image replaces x0_hat.
            with contextlib.nullcontext() if use_dino else torch.no_grad():
                image = self.decode(x0)
                dino, fixed, pointed, predicted_boxes = self.detail_loss(row, image, maps)
            weight = self.args.dino_weight if use_dino else 0.
            loss = denoise + self.args.spatial_weight * localization + weight * dino
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite objective at " + name + "/" + str(step))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.projections + list(self.tracker.parameters()), 1.)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            record = {"arm": name, "step": step + 1, "id": row["row"]["id"], "timestep": t,
                      "t_fraction": t / (self.steps - 1), "loss_total": float(loss.detach()),
                      "loss_denoise": float(denoise.detach()), "loss_spatial": float(localization.detach()),
                      "dino_distance": float(dino.detach()), "dino_weight": weight,
                      "gradient_norm_before_clip": float(grad_norm),
                      "elapsed_seconds": time.time() - start,
                      "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30}
            with log.open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            atomic(self.args.output / "PROGRESS.json", record)
            if step == 0 or (step + 1) % 10 == 0:
                print("TRAIN", json.dumps(record), flush=True)
            del pred, noise, x0, maps, image, dino, loss, denoise, localization, fixed, pointed, predicted_boxes
            self.tracker.clear()
            if (step + 1) % 100 == 0 or step + 1 == self.args.steps:
                checkpoint = {"arm": name, "step": step + 1,
                              "projection_parameters": [p.detach().cpu() for p in self.projections],
                              "component_keys": {k: v.detach().cpu() for k, v in self.tracker.state_dict().items()},
                              "layers": LAYERS, "concepts": CONCEPTS, "optimizer": optimizer.state_dict()}
                torch.save(checkpoint, self.args.output / (name + "_latest.pt"))
                if step + 1 in {100, self.args.steps}:
                    self.evaluate(name + "_" + str(step + 1), save=step + 1 == self.args.steps)
        unchanged = torch.equal(self.evaluator.dino.embeddings.cls_token.detach(), self.initial_dino)
        if not unchanged or any(p.grad is not None for p in self.evaluator.parameters()):
            raise AssertionError("Frozen DINO acquired parameter gradients or changed weights")
        return self.evaluate(name + "_final", timesteps=(.05, .1, .25, .5, .75, .9, .95), save=True)


def cpu_checks():
    boxes = torch.tensor([[.25, .25, .5, .75]])
    uniform = torch.ones(1, 8, 12)
    expected = (.5 - .25) * (.75 - .25)
    assert abs(float(spatial_mass(uniform, boxes)) - expected) < 1e-6
    for bad in [[[0, 0, 0, 1]], [[-.1, 0, 1, 1]], [[0, 0, 1.1, 1]]]:
        try:
            box_coverage(torch.tensor(bad), (8, 12))
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid box accepted")
    image = torch.rand(1, 3, 18, 30, requires_grad=True)
    roi = torch.tensor([[[.23, .21, .71, .83]]], requires_grad=True)
    crop_regions(image, roi, 12).square().mean().backward()
    assert image.grad.abs().sum() > 0 and roi.grad.abs().sum() > 0
    alpha = torch.tensor([.75, .2]);t = torch.tensor([1])
    z = torch.randn(1, 4, 3, 2);noise = torch.randn_like(z)
    noisy = alpha[t].sqrt() * z + (1 - alpha[t]).sqrt() * noise
    assert torch.allclose(reconstruct_x0(noisy, noise, alpha, t, "epsilon"), z, atol=1e-6)
    print("CPU_CONTRACTS_PASSED", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-checks", action="store_true")
    p.add_argument("--data", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--leffa-code", type=Path)
    p.add_argument("--leffa-weights", type=Path)
    p.add_argument("--dino-weights", type=Path)
    p.add_argument("--steps", type=int, default=500, help="Updates per matched arm; default totals 1000")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--query-lr", type=float, default=.01)
    p.add_argument("--spatial-weight", type=float, default=5.)
    p.add_argument("--dino-weight", type=float, default=.1)
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    cpu_checks()
    if args.cpu_checks:
        return
    if not all((args.data, args.output, args.leffa_code, args.leffa_weights, args.dino_weights)):
        p.error("GPU execution requires data, output, LeFFA code/weights and DINO weights")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name():
        raise RuntimeError("This experiment is authorized for H200")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    manifest = json.loads((args.data / "manifest.json").read_text())
    groups = {}
    for row in manifest["records"]:
        if groups.setdefault(row["garment_id"], row["split"]) != row["split"]:
            raise ValueError("Garment leakage between training and validation")
    meta = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "gpu": torch.cuda.get_device_name(), "gpu_free_total_bytes": list(torch.cuda.mem_get_info()),
            "torch": torch.__version__, "code_sha256": sha(__file__),
            "manifest_sha256": sha(args.data / "manifest.json"),
            "annotation_regime": manifest["label_source"],
            "claims": "Single-timestep denoising estimates, not full diffusion samples; learned semantic readout, not native text attention."}
    atomic(args.output / "RUN.json", meta)
    try:
        experiment = Experiment(args, manifest)
        experiment.verify()
        if args.verify_only:
            return
        baseline = experiment.evaluate("baseline", timesteps=(.05, .1, .25, .5, .75, .9, .95), save=True)
        control = experiment.train_arm("denoise_spatial_control", False)
        joint = experiment.train_arm("denoise_spatial_dino", True)
        masses = joint["mass_at_half"]
        criteria = {
            "localization_over_80pct_each_component_at_half": all(v is not None and v > .8 for v in masses.values()),
            "fixed_roi_dino_improves_over_initial": joint["mean_fixed_roi_dino_distance"] < baseline["mean_fixed_roi_dino_distance"],
            "fixed_roi_dino_improves_over_matched_control": joint["mean_fixed_roi_dino_distance"] < control["mean_fixed_roi_dino_distance"],
            "pointed_roi_dino_improves_over_matched_control": joint["mean_pointed_roi_dino_distance"] < control["mean_pointed_roi_dino_distance"],
            "dino_gradient_path_verified": True,
            "completed_at_least_1000_iterations_without_oom": args.steps * 2 >= 1000 and experiment.oom_count == 0}
        result = {"status": "SUCCESSFUL" if all(criteria.values()) else "COMPLETED_CRITERIA_NOT_ALL_MET",
                  "criteria": criteria, "iterations": args.steps * 2, "oom_count": experiment.oom_count,
                  "baseline": {k: v for k, v in baseline.items() if k != "records"},
                  "control": {k: v for k, v in control.items() if k != "records"},
                  "joint": {k: v for k, v in joint.items() if k != "records"},
                  "forward_calls": experiment.calls, "annotation_caveat": manifest["label_source"]}
        atomic(args.output / "RESULT.json", result)
        print("EXPERIMENT_COMPLETE", json.dumps(result), flush=True)
    except Exception as exc:
        atomic(args.output / "FAILED.json", {"error": str(exc), "traceback": traceback.format_exc(),
                                              "oom": isinstance(exc, torch.cuda.OutOfMemoryError)})
        raise


if __name__ == "__main__":
    main()
