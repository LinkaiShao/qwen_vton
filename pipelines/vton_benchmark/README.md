# HR-VITON warping benchmark code

Inference, frozen regional scoring and report generation for the ninth model in the VTON comparison. The original 128 cases, eight model outputs and score records are reused. HR-VITON runs with both the shared benchmark mask and its native masking; intermediate RGB warps and flow fields are saved.

[Published comparison](../../docs/warp-benchmark/README.md) · [Live website](https://linkaishao.github.io/qwen_vton/warp-benchmark/) · [Exact run provenance](run_reference/provenance.json)

The inference and scoring modules are the code used for the completed run. This package includes their Python support modules. It requires the VITON-HD dataset and the original benchmark run directory; dataset images, pretrained weights and the custom trained expert checkpoint are not bundled. `bootstrap.py` fetches the official public sources and HR-VITON weights with pinned revisions and SHA256 verification.

## Setup

Run from this directory. The verified environment used Python 3.12, PyTorch 2.10.0 with CUDA 12.8, torchvision 0.25.0 and an RTX 5090.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python bootstrap.py --job /path/to/new-hrviton-run
```

HR-VITON source: `sangyun884/HR-VITON`, revision `2715bdd687b3a07b8b8bcb3e44aa5533b28c1f15`. OOTDiffusion supplies the shared benchmark mask utility, pinned to `13ef0faba266cdde9febc8ad39be2395bbb89d9c`. The bootstrap keeps third-party source and weights outside Git tracking. Their original licenses apply; HR-VITON is CC BY-NC 4.0.

## Required inputs

`--dataset` points to VITON-HD with `test/image`, `cloth`, `cloth-mask`, `image-parse-v3`, `image-parse-agnostic-v3.2`, `image-densepose`, `openpose_img` and `openpose_json`.

`--source` points to the completed original benchmark directory, containing:

```text
experts.pt                         frozen custom expert checkpoint
collected/manifest.jsonl
collected/scores/{model}/{id}.json  all eight original models
collected/grounding/*.json          original Molmo GT receipts
collected/predictions/leffa/*.png   original PNGs for scorer checks and gallery
collected/report/summary.json
```

The scorer enforces expert SHA256 `88373a448b4bf91d47d18b068e8872043489a042321272f659ebd4eaf4e22fc6` through the source code's `EXPERT_SHA` constant; consult [scorer validation](run_reference/scorer_validation.json) for the run receipt. The exact 128-case cohort is in `run_reference/manifest.jsonl`.

## Run inference and scoring

Use explicit paths; the original scripts retain this workstation's defaults. Select a free GPU with `CUDA_VISIBLE_DEVICES`.

```bash
python -m garment_structure_eval.benchmark.hrviton_extension prepare \
  --dataset /path/to/VITON-HD-dataset \
  --source /path/to/original-eight-model-run \
  --job /path/to/new-hrviton-run

CUDA_VISIBLE_DEVICES=0 python -m garment_structure_eval.benchmark.hrviton_extension infer \
  --dataset /path/to/VITON-HD-dataset \
  --source /path/to/original-eight-model-run \
  --job /path/to/new-hrviton-run

CUDA_VISIBLE_DEVICES=0 python -m garment_structure_eval.benchmark.hrviton_extension score \
  --dataset /path/to/VITON-HD-dataset \
  --source /path/to/original-eight-model-run \
  --job /path/to/new-hrviton-run
```

Use `infer --limit 2` for a smoke run. Re-running inference resumes verified existing outputs. `raw_warp/` contains the RGB resampling before occlusion handling, `warp/` the garment supplied to the generator, and `flows/` the learned displacement fields. `predictions/hrviton/` and `predictions/hrviton_native/` hold final PNGs.

The scorer reuses original GT Molmo points, verifies four prior LeFFA scores, and scores each variant with the frozen DINO experts. It does not run new prediction localization. This is inference only; it does not train any model.

## Rebuild the report

Run this from a checkout containing the existing `docs/region-benchmark` assets. Keep the new report next to that directory so the publisher can reuse the original seven example selections.

```bash
python -m garment_structure_eval.benchmark.publish_hrviton \
  --dataset /path/to/VITON-HD-dataset \
  --source /path/to/original-eight-model-run \
  --job /path/to/new-hrviton-run \
  --site ../../docs/warp-benchmark
```

This produces nine-model heatmap, bar and line charts; the separate native-mask sensitivity table; all 128 comparison sheets; and HTML/Markdown reports. It writes local files and does not push to GitHub automatically.

The scorer uses full-image 224×224 DINO features and 5×5 token windows. Its feature distances are not validated as tiny-detail accuracy. The report retains coverage and automatic-label limitations from the original comparison.
