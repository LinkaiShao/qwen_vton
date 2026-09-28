# Single-timestep localization and DINO feedback in LeFFA

This experiment asks whether LeFFA can expose a component location and receive a
local image-feature gradient during a normal random-timestep training forward.
It does not run a multi-step denoising sampler or call Molmo during training.

## Necessary corrections to the original specification

- The official LeFFA model disables text `attn2` with `SkipAttnProcessor`. Image
  conditioning happens in `attn1`, over concatenated person and garment tokens.
  Attaching a hook to an Attention module captures output features, not softmax
  attention probabilities. See the [official implementation](https://github.com/franciszzj/Leffa/blob/main/leffa/model.py)
  and [LeFFA paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_Learning_Flow_Fields_in_Attention_for_Controllable_Person_Image_Generation_CVPR_2025_paper.pdf).
- This checkpoint uses a DDPM epsilon predictor, not a flow velocity model.
  Reconstruction is `x0_hat = (x_t - sqrt(1-alpha_bar_t)*eps_pred) /
  sqrt(alpha_bar_t)`. The script also implements the different v-prediction
  formula and tests reconstruction numerically.
- DINO consumes decoded RGB, not interpolated four-channel VAE latents. Predicted
  crops retain autograd through the frozen DINO and frozen VAE. Only target
  features and the reference encoder use `no_grad`.
- A random-tensor smoke test cannot establish localization, image fidelity,
  parameter updates, or training stability. This runner uses real paired images,
  pretrained weights, gradients, optimization steps, and a validation split.

## Implemented localization readout

Three learned component keys represent collar/neckline, image-left sleeve end,
image-right sleeve end. At two existing up-block image-attention layers, forward
hooks capture the actual Q projection outputs. Each component key passes through
that layer's existing K projection. Dot products with the target half of the Q
field are normalized spatially, averaged over heads and aligned to the finest
rectangular attention grid. Component maps receive the requested inside-box mass
loss. Annotation coordinates are never inputs to the attention logits or the
LeFFA forward pass.

This is an added semantic readout of live LeFFA features. It is **not** a claim
that pretrained LeFFA already contains language tokens for collars/cuffs, nor a
claim that the original image-to-image attention weights are semantic text maps.
Installing hooks must leave denoiser output bit-identical before optimization.
The pretrained network remains intact except for updates to the selected real
Q/K projections. A separate key embedding per component is trained jointly.

Each example requires exactly one frozen garment-reference encoder pass and one
generative UNet pass at the sampled timestep. VAE/DINO operations are additional
work, but there is no denoising trajectory or full image generation before loss.

## Data and labels

The initial cohort is the 32 VITON-HD **training-split** images with existing
cached collar/sleeve-end proposals: 24 optimization images and 8 validation
images, separated by garment image identity. The annotations originated from a
previous MolmoPoint job; **no Molmo is run in this experiment**. They are automatic
pseudo-labels, not human ground truth. Collar here includes neckline; cuff here
includes a short sleeve's end. All images with usable cached proposals are kept.
There is no manual mask correction or per-case attention rule.

Duplicate nearby sleeve proposals are merged; distinct image-left and image-right
ends get different component IDs. Multiple collar points are enclosed together.
Image inputs are 512x384; coordinates remain normalized. The agnostic mask uses
the existing OOTD recipe from VITON parsing and pose, with explicit LIP-to-ATR
mapping. Source file hashes and native dimensions are retained in the manifest.
Input VAE encodings use deterministic posterior means for matched comparisons.

## Objectives and controls

Two arms start from identical model/readout parameters and use the same 500
minibatch indices, sampled timesteps, and noise seeds:

1. Control: epsilon MSE + `5 * (1 - inside_mass)`.
2. Joint: the control objective + `0.1 * DINO_crop_distance`.

DINO distance averages (a) crops at the labelled detail site and (b) crops at the
readout's differentiable attention center, each compared with the corresponding
real target crop. The second path sends gradients to the component keys as well
as the generative projections. Crop widths/heights are supplied by training
metadata; the experiment learns centers, not component extents. Frozen DINOv2
ViT-B/14 CLS features use ImageNet normalization and cosine distance.

All 1,000 training iterations decode and evaluate DINO crops. The control computes
that score without backpropagating it. No model is called to localize a prediction.
A fixed-noise validation sweep covers t fractions .05, .1, .25, .5, .75, .9, .95.
Cross-image shuffled heatmaps are also scored to expose reliance on a generic
layout prior. Same-component before/after crops and attention maps are saved.

## Success is measured, not assumed

The runner writes `SUCCESSFUL` only if every explicit criterion is satisfied:

- Mean inside-box mass exceeds .80 for each component at t=.5 on validation.
- DINO gradients reach existing Q/K projections and the component keys.
- Fixed-site DINO distance beats both the initial model and the matched control.
- Attention-centered DINO distance also beats the matched control.
- At least 1,000 iterations complete without OOM.

Otherwise the completed measurements say `COMPLETED_CRITERIA_NOT_ALL_MET`. An
execution exception writes `FAILED.json`. Even success establishes only this
small pseudo-labelled, single-timestep experiment; it does not establish improved
fully sampled VTON outputs or localization at every possible timestep.

## Running

`verify_vton_localization.py` is self-contained apart from the official LeFFA
source package, PyTorch/diffusers/transformers, pretrained weights, and the paired
image manifest. Its CPU numerical contracts need no downloaded models:

```bash
python verify_vton_localization.py --cpu-checks
```

A full run on the authorized H200:

```bash
python verify_vton_localization.py \
  --data /path/to/prepared_dataset \
  --output /path/to/new_output \
  --leffa-code /path/to/Leffa \
  --leffa-weights /path/to/Leffa_model_snapshot \
  --dino-weights /path/to/dinov2-base_snapshot \
  --steps 500
```

Use `--verify-only` for the real forward/backward integration check. The runner
refuses an occupied output directory. The workspace-only `remote.py` controller
uses the existing ICRN token file without embedding credentials and keeps a
notebook kernel busy throughout the remote subprocess. That controller depends on
the workspace's ICRN client; it is not required for the portable runner above.

Pinned assets already available on H200:

- LeFFA source: `05a259104b6927607776c7edb3e86b75406f20aa`.
- `franciszzj/Leffa`: `61d3390f444506f052feedb0b243cd5369c29c89`.
- `facebook/dinov2-base`: `f9e44c814b77203eaa57a6bdbbd535f21ede1415`.

Outputs include `RUN.json`, gradient verification, per-step JSONL logs, checkpoints,
fixed-seed evaluations, raw attention arrays, image estimates, and the final
criterion-by-criterion result. The original retrieval PDF motivates component
scoring; it does not itself validate this proposed replacement for grounding.

## Reproducing the data

The committed `manifest.json` records all 32 examples, source image hashes, original
automatic proposals and the garment-separated split. The original data-preparation
command was independently reproduced: all 128 prepared image files were byte-identical.

```bash
python prepare_data.py \
  --dataset /path/to/VITON-HD-dataset/train \
  --manifest manifest.json \
  --ootd-utils /path/to/OOTDiffusion/run/utils_ootd.py \
  --output /path/to/empty/prepared_dataset
```

OOTDiffusion source revision: `13ef0faba266cdde9febc8ad39be2395bbb89d9c`.
Preparation requires NumPy, Pillow and OpenCV. The separate reporting script
requires Matplotlib. `requirements-h200.txt` records the observed training environment;
it does not include unrelated packages from the shared server. Download the official
LeFFA source/weights and DINO weights at the pinned revisions listed above.

## Recorded first experiment and exploratory follow-up

The original 500-update-per-arm run completed 1,000 updates with zero OOMs. The joint
arm attained 67.48% collar, 50.47% left sleeve and 54.08% right sleeve box mass at
t=.5. Its fixed-crop DINO distance was 0.26394 versus 0.26254 for the matched control
(lower is better). Therefore it **failed** both the 80% localization target and the
test that DINO feedback adds fixed-site detail improvement. The gradient-path test
passed. This result must not be described as a quality success.

The follow-up uses `--aggregation full_logits --dino-weight 1 --pointed-mix .1
--steps 1500`. Full-logit aggregation computes a dot product across all Q/K head
dimensions before one spatial softmax, instead of averaging separately normalized
head maps. DINO weights fixed-site crops by .9 and attention-centered crops by .1.
The 500-update original remains reproducible with the defaults.

This is an exploratory follow-up: it changes aggregation, loss weighting and duration
together after seeing the eight validation images. Those images are now a development
set, not an untouched final test set. Comparisons cannot identify which of the three
changes caused a difference. A future generalization claim needs new unseen garments
and an independent evaluation, including fully sampled outputs if claiming try-on
quality.

The follow-up completed 3,000 updates with zero OOMs. Fixed-crop DINO distance was
**0.23848 versus 0.26604 for its matched control (10.36% lower)**. Attention-centered
distance was 0.32133 versus 0.36100. At t=.5, joint localization mass was **74.63%
collar, 74.83% left sleeve, 81.65% right sleeve**. It therefore still fails the full
80%-per-component localization criterion. DINO improved the fixed-site feature
score while reducing localization mass compared with its control (74.89%, 86.14%,
83.99%). Both successes and failures are retained in the public report.

To build the static report from downloaded experiment artifacts:

```bash
python build_report.py --run /path/to/run --data /path/to/prepared_dataset \
  --output /path/to/report
```

All eight development examples appear in the report, with the real target, initial
single-step estimate, localization control, joint result, fixed detail crops and
heatmaps. Feature-score changes alone are not labelled as visually fixed defects.

## Separate full-sample check

`evaluate_samples.py` loads the final checkpoints and runs the official LeFFA
pipeline for 20 DDPM steps at guidance 2.5 on every development image. Initial,
control and joint models use identical per-image seeds at 512x384, without final
RGB repaint. This is a **post-training evaluation**; these trajectories were never
part of a training loss. Fixed-region DINO scores and full images are saved for
all three variants. `build_sampling_report.py` renders the matched comparisons.

Observed full-sample fixed-crop distance: initial 0.24101, control 0.23798, joint
0.22804 (4.18% lower than control). **Transfer was mixed**: collar distance improved
from 0.20144 to 0.16599, but left sleeve worsened from 0.34566 to 0.35905 and right
sleeve from 0.22773 to 0.26246. The image-bootstrap 95% interval for joint minus
control was [-0.08264, +0.00622], n=8, so the overall gain is not conclusive. Some
missing trim details remain visibly missing despite better DINO scores.

```bash
python evaluate_samples.py --data /path/to/prepared_dataset \
  --run /path/to/followup_run --output /path/to/new_sample_output \
  --leffa-code /path/to/Leffa --leffa-weights /path/to/Leffa_model_snapshot \
  --dino-weights /path/to/dinov2-base_snapshot
```

All training images and model weights remain external assets. The portable code
and manifest are sufficient to rebuild inputs from the original VITON-HD data.
Checkpoints and full training logs are retained in the experiment directories;
the public report includes measured results, raw region scores and update audits.
