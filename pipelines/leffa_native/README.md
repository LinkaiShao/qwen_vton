# Native LeFFA attention + dense DINO training

This is the replacement for the earlier independent component-key readout.
There are **no added localization parameters, named part boxes, or Molmo calls**.
The target/reference attention block is recomputed from the exact projected Q/K
used by LeFFA's existing fused attention. Its full target+reference denominator
is preserved, and observation does not change the denoiser output.

## Data and losses

The prepared manifest contains 11,135 training, 512 development and 256 official
test examples. Splits are deterministic and grouped by byte-identical garment
image SHA256; test garments with an identical training hash are excluded.
Near-duplicate visual images are not guaranteed absent. Existing visible-garment
parsing masks are distinct from the larger OOTD inpainting masks.

Frozen DINOv2-B/14 sees RGB at 448x336, giving a 32x24 grid. Clean product/target
features supply automatic correspondences with cosine >= .5 and bidirectional
cycle error <= one cell. These are pseudo-labels, not ground truth. Empty parsing
or reliable-match sets disable their corresponding auxiliary term, never create
fake matches. Such cases remain in diffusion training and are flagged in logs.

The native reference-attention block is averaged across heads and reduced to the
DINO grid by summing reference-key cells and averaging person-query cells. The
two traced layers are the final attention blocks in up_blocks.2 and up_blocks.3.
Correspondence CE supervises the conditional reference distribution against a
one-cell Gaussian around each reliable match. Absolute reference mass is retained
separately; conditional normalization is not described as the actual AV weight.

The checkpoint predicts epsilon. Its one-forward clean latent estimate is
`(z_t - sqrt(1-alpha_bar)*epsilon_pred)/sqrt(alpha_bar)`. Frozen VAE decoding and
DINO retain derivatives to predicted pixels. Dense cosine loss compares identical
target coordinates: 50% uniform visible-garment coverage and 50% equal-source-patch
coverage derived from native attention. Attention weighting is detached only in
this image loss to prevent moving the scoring window to evade errors.

The objective is epsilon MSE + .1 correspondence CE + w(t) DINO, with
`w(t)=min(1,sqrt(SNR(t)))`, positive at every actual scheduler timestep. This weight
bounds the explicit epsilon-to-x0 error amplification; it does not guarantee
high-noise image fidelity. Q/K/V/output projections in the last two generative
up-blocks train at lr=1e-5, AdamW decay=.01, clip=1; all other weights are frozen.

## Four controlled arms

`diffusion`, `correspondence`, `dino_all`, `dino_low` each run 3,000 optimizer
updates, microbatch 1, accumulation 4, from identical weights and the same example,
timestep and noise schedule. The low-noise arm applies DINO only at t<500.
Auxiliary scores are still computed without their image gradients in controls.
Checkpoints every 250 updates allow deterministic replay from the last save.

## Verification and evaluation

`test_core.py` runs meaningful CPU contracts for full-softmax probability/gradient
equivalence, spatial reduction, observation parity, native value interventions,
fixed-coordinate loss gradients, rejected matches and 1,000-step algebra.
`runner.py verify` checks real H200 LeFFA output parity and native Q/K/V/output
gradients. `runner.py sweep` tests all 1,000 scheduler timesteps on 16 rotating real
development examples. Full sampling is performed ONLY by `evaluate.py`.

Final evaluation uses 256 fixed test pairs, 20 DDPM steps, CFG 2.5, matched seeds,
no final RGB repaint, DINO, independent LPIPS and SSIM. Bootstrap intervals are by
image. Controlled zeroing of selected native reference values measures whether
predicted footprints localize output changes better than equal-area shifted
controls, reported in five noise bands. This does not imply reference features
contain only local pixels: encoder features are contextual.

`SUCCESSFUL` requires native gradient checks, the timestep sweep, causal advantage
in every noise band, and statistically supported DINO AND garment LPIPS gains
against diffusion-only and correspondence-only controls. Otherwise a completed
run is explicitly `COMPLETED_CRITERIA_NOT_ALL_MET`.

The HTML report shows the biggest improvements, biggest regressions and a fixed
sample. Input, target and four model outputs use identical detail crops. Clicking
any source patch displays the *actual* native attention from a separate single
timestep forward; these estimates are clearly distinguished from full samples.

## Execution

Use the existing pinned LeFFA/DINO assets and H200 environment recorded by the
adjacent `leffa_single_step` experiment. The new pipeline is self-contained apart
from official LeFFA, the OOTD mask helper, pretrained assets and dataset.

```
python test_core.py
python prepare.py --dataset /path/to/VITON-HD-dataset --output /path/to/job \
  --ootd-utils /path/to/OOTDiffusion/run/utils_ootd.py
python runner.py cache --data /path/to/job --cache /path/to/job/cache \
  --output /path/to/job/run --leffa-code /path/to/Leffa \
  --leffa-weights /path/to/Leffa_snapshot --dino-weights /path/to/DINO_snapshot
```

The same path arguments apply to `verify`, `sweep`, `train`, and `evaluate.py`.
`supervise.py` runs all phases and reporting, waits for 24 GiB of free shared H200
memory, and stops its own worker at a 23-hour session deadline. It never kills
another user's process. `remote.py` uses the private existing ICRN credential file
without printing or copying credentials into experiment artifacts.

Research basis: [LeFFA](https://arxiv.org/html/2412.08486v2),
[CORAL](https://arxiv.org/html/2602.17636v1),
[SIFT-VTON](https://arxiv.org/html/2605.01296v1), and
[PixelGen](https://arxiv.org/html/2602.02493v1). This combines/adapts ideas; it is not
an exact reproduction of CORAL (DiT/DINOv3) or SIFT-VTON (SIFT/StableVITON).
