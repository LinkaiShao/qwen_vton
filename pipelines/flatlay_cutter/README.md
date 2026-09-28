# Flatlay-reference worn-garment cutter

Use a **product-only photograph to identify the garment**, then cut that garment from a **photo of a person wearing it**. The transparent PNG contains the original worn-photo pixels. It does not generate a replacement garment or paste flatlay pixels into the worn image.

[Reviewed before/after gallery](https://linkaishao.github.io/qwen_vton/flatlay-cutter/repair/) · [Recorded evaluation](evaluation_summary.json) · [Model/source provenance](provenance.json)

This is the portable implementation of the reviewed September 27, 2026 pipeline. Its SAM3 inference, component feature matching, colour filter, and bounded noun retries are unchanged. Workstation paths, remote-service imports, and the machine-specific GPU assertion have been removed. The original run reused cached automatic reference masks; the package accepts those explicitly or computes new masks from the supplied references.

## Install

Use Python 3.11 or newer and a CUDA-enabled PyTorch installation. The measured environment used PyTorch 2.10.0 with CUDA 12.8, torchvision 0.25.0, and Transformers 5.2.0. Dependency versions are recorded in `pyproject.toml`.

```bash
git clone https://github.com/LinkaiShao/qwen_vton.git
cd qwen_vton/pipelines/flatlay_cutter
python -m venv .venv
source .venv/bin/activate
pip install -e .
flatlay-cut --help
```

CUDA and BF16 support are required. The portable package is validated on an RTX 5090; an H200 can use the same CUDA code path, but this package has not been benchmarked on H200. Choose the intended GPU with `CUDA_VISIBLE_DEVICES`; the runner uses its first visible CUDA device. Automatic preparation and cutting load their models sequentially.

Model weights are **not in this repository**. The runner downloads these exact revisions into your Hugging Face cache, using your existing model access/authentication:

| Stage | Model | Pinned revision |
|---|---|---|
| Segmentation and reference features | `facebook/sam3` | `3c879f39826c281e95690f02c7821c4de09afae7` |
| Automatic photo selection, framing, reference naming | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |

Use `--offline` when these revisions are already cached. Obtain model access through the respective model provider when required. Existing model licenses continue to apply.

## Run on an SKU's photographs

Create a manifest like [`examples/raw_sku.json`](examples/raw_sku.json), listing the photographs belonging to each SKU. Image paths are relative to the manifest directory, unless `--data-root` is specified. Images and masks are not bundled with these example manifests.

```json
{
  "skus": [{
    "key": "my_product",
    "views": [
      {"id": "product", "source": "images/product.jpg"},
      {"id": "front", "source": "images/worn_front.jpg"},
      {"id": "back", "source": "images/worn_back.jpg"}
    ]
  }]
}
```

Run automatic preparation and cutting:

```bash
CUDA_VISIBLE_DEVICES=0 flatlay-cut run \
  --manifest /path/to/sku_images.json \
  --output /path/to/new_run
```

Preparation automatically identifies complete product-only references, names their visible product types, and excludes worn detail closeups. Flatlays, hanger shots, and invisible-mannequin product shots qualify. An SKU with no qualifying reference is explicitly skipped. No human mask annotation is required.

Each worn overview is cut independently. A visible portion can be kept even when the garment is partly hidden by another garment. Product titles, other worn views, and annotation masks never determine membership. The framing check uses the product reference plus the one candidate target; it is setup work, outside the fast cutting loop.

Every output directory must be new or empty, so reruns cannot silently overwrite an earlier result. A processing error or empty predicted mask makes the command exit with status 2 and is recorded in `RESULTS.json`; missing-reference or no-overview skips are also explicit there.

## Run only the fast cutter

If an upstream automatic selector already identifies the product-only references and worn overview targets, use [`examples/selected_sku.json`](examples/selected_sku.json). Supplying these fields bypasses photo-role selection; the caller must exclude closeups. Missing reference nouns are inferred automatically during `prepare`/`run`.

For repeated use, prepare once and pass the resulting manifest to the SAM3-only command:

```bash
CUDA_VISIBLE_DEVICES=0 flatlay-cut prepare \
  --manifest /path/to/selected_sku.json \
  --output /path/to/setup_run

CUDA_VISIBLE_DEVICES=0 flatlay-cut cut \
  --manifest /path/to/setup_run/prepared_manifest.json \
  --output /path/to/cut_run
```

`cut` makes **zero VLM calls**. `reference_queries` must contain specific nouns already derived from the product image. Generic names such as `clothing` are rejected in the worn-target path. Multiple references and multiple worn targets per SKU are supported, and reference feature banks are reused across that SKU's targets.

Optional reference fields for replaying a previous automatic setup:

```json
{
  "id": "product",
  "source": "images/product.jpg",
  "reference_queries": ["shirt"],
  "mask": "reference_masks/product.png",
  "source_sha256": "SHA256_OF_SOURCE_FILE",
  "mask_sha256": "SHA256_OF_MASK_FILE"
}
```

Omit `mask` and its hash to segment the reference automatically. Supplied masks must be nonempty, match the EXIF-corrected source dimensions, and represent the product in the product-only photo. Hashes are checked when supplied. Replaying the gallery exactly requires its original reference images, nouns, and automatic reference masks; regenerating setup can change the result.

For integration into a long-running worker, prepare references once in memory:

```python
from flatlay_cutter.engine import ReferenceCutter
from flatlay_cutter.io import export_cutout, read_image

cutter = ReferenceCutter(local_files_only=True)
sku = {
    "key": "my_product",
    "references": [{
        "id": "product",
        "source": "/data/product.jpg",
        "reference_queries": ["shirt"],  # From automatic reference preparation.
    }],
}
cutter.prepare(sku)
image = read_image("/data/worn.jpg")
mask, diagnostics = cutter.cut(image, "my_product")
if mask.any():
    export_cutout(image, mask, "/data/result")
```

## Outputs

```text
new_run/
  prepared_manifest.json       # run/prepare: selected references, roles, nouns
  setup_cache/                 # run/prepare: automatic setup evidence
  RESULTS.json                # all output/error/skip records and code hashes
  outputs/<sku>/<target>/
    mask.png                  # binary mask at full worn-image resolution
    cutout.png                # transparent PNG cropped around retained pixels
    crop.jpg                  # rectangular worn-image crop, without transparency
    result.json               # candidates, matching scores, prompt attempts, hashes
```

For `empty_mask`, only the zero mask and diagnostics are exported. `prepare` produces setup files only. EXIF orientation is applied before segmentation; JPEG, PNG, WebP, and AVIF inputs are supported. Alpha is binary: this is segmentation, not physical transparency matting or reconstruction of hidden fabric.

## What the cutter does

1. Encode each product-only reference with frozen SAM3. Build colour support and spatial feature banks separately for its visible components. Broad `clothing` prompting is restricted to the isolated product to cover attached contrasting layers.
2. Encode the worn target once. Specific product nouns produce candidate masks.
3. Compare each candidate with reference foreground and background features, component by component, with a colour prefilter. Keep the union of compatible whole candidates.
4. If a noun yields no accepted candidate, try at most two related nouns using the same image encoding. A retry must overlap an original strong spatial hypothesis by at least 0.80 IoU. This prevents a broad retry from freely selecting another garment.
5. Export native SAM3 boundaries and original worn-image RGB pixels.

There is no training and no Molmo/DINO model in this version. Qwen handles automatic setup; SAM3 supplies both candidate masks and the visual features used for selection.

## Validation and speed

The reviewed development batch contains 190 selected worn targets. All produced nonempty cutouts; mean annotation IoU was 0.8853 before and 0.9103 after the repair. The prior 33-image sample remained pixel-identical, and four development probes using unrelated product references rejected. These measurements do not establish perfect segmentation or general rejection of every similar garment.

On RTX 5090, paired warm mask inference averaged **90.1 ms**, with **116.8 ms p95**. The 190-target run averaged **320 ms from file read through export**. Model loading, automatic photo selection/naming, and per-SKU reference preparation are excluded. CLI startup is therefore much slower than the warm per-image figure; use a persistent worker for low latency.

Known errors include missing collar portions, fine cords, and ragged hems. Existing annotations can also include unrelated garments or omit real attached components. The gallery exposes the flatlay, worn image, and actual output so that these differences can be inspected directly.

Run the CPU contract checks with:

```bash
python -m unittest discover -s tests -v
```

`port_validation.json` records the separate GPU parity check of this package against the reviewed pipeline. No training datasets, model weights, service credentials, or machine-specific launch configuration are shipped.
