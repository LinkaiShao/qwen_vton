# LeFFA: localization before appearance training

This is a frozen-generator audit. There is no optimizer, Molmo, DINO model, or appearance fine-tuning entry point. Passing the audit does not automatically start training.

The question is whether the generator's own attention can locate the source detail it uses, at one sampled training timestep. Attention weights alone are not sufficient evidence.

## What runs

1. Qwen names the isolated reference garment. The existing flatlay cutter selects that garment on the person. Original LIP body/other-garment regions are protected; identity and inpainting masks are separate. An empty/conflicting mask fails explicitly; the old mask is never a fallback.
2. Frozen SAM3 locates source and evaluation details using two agreeing prompts per concept. Undetected or disagreeing details are **uncertain**, not assumed absent. Source masks are cached once. No VLM generates coordinates: an initial Qwen bounding-box attempt failed visual inspection and was retired before probing.
3. Hooks copy the actual generator `attn1.to_q` and `attn1.to_k` outputs in six late up-block attention layers. LeFFA concatenates person tokens followed by reference tokens. The readout preserves the complete person+reference softmax denominator, then sums reference attention over the marked source region. It does not train an independent location head.
4. Retain individual heads and layer means (55 candidates). Choose one fixed candidate per detail using 64 development images; freeze it before 256 fresh held-out garment groups. Eight previous failures are displayed separately as regressions. Additional cuff/sleeve-end instance readouts remain separate; their confidence is explicitly uncalibrated and their boxes must not be merged into a torso-wide DINO crop.
5. At each timestep the live trace uses one reference UNet and one generator UNet forward. DDPM epsilon prediction reconstructs `x0 = (xt - sqrt(1-alpha_bar)*epsilon) / sqrt(alpha_bar)`, followed by the real VAE decoder. It is not a completed 20-step sample. No flow-matching reconstruction formula is substituted.
6. Offline causal tests remove/increase the selected source V region in the chosen head/layer, compare the changed prediction against the same-noise baseline, and include repeat, compact equal-area unrelated-source, and shifted equal-area destination controls. These extra forwards are evaluation only.

Source labels identify **what** region to trace. Target/generated-image SAM3 masks only score the predicted marks; they cannot move them or set live confidence. Automatic annotations remain imperfect proxies, with correlated teacher biases. Source presence, raw attention concentration and a frozen development threshold determine whether the live mark is accepted.

## Resolution and gates

The finest grid is 64×48 on a 512×384 image: eight pixels per cell. Earlier selected layers are coarser and are interpolated only for display. No pixel-exact localization claim is made.

Every detail must pass every one of five timestep bands: at least 30 visible held-out images; 80% coverage; 90% peak hits within an eight-pixel tolerance; 80% inside mass; equivalent actual-generated-image localization; and positive 95% bootstrap intervals for causal localization and source specificity. The audit also rotates 16 development images through all 1,000 actual scheduler timesteps. Finite maps are a stability check, not an accuracy result. Failure blocks appearance training.

## Local execution

Workspace: `/home/link/Desktop/Code/fashion_gen_testing`.
Artifacts: `/mnt/nvme0/leffa_native/localization_v2`.
GPU: RTX 5090 UUID `GPU-dbb7940e-7461-387e-c0c2-a2e33f9b78ac`; the RTX PRO 6000 is not used.

Two pinned environments are required because SAM3 and LeFFA use different transformers versions:

```bash
/mnt/nvme0/leffa_native/env/bin/python garment_structure_eval/leffa_localization/test_contracts.py
/mnt/nvme0/leffa_native/env/bin/python garment_structure_eval/leffa_localization/data.py
/mnt/nvme0/leffa_native/env/bin/python garment_structure_eval/leffa_localization/run.py --max-hours 9
```

`run.py` publishes progress and final evidence to the existing GitHub Pages report. `--no-publish` disables publication. It records owned PIDs, preserves logs, has a maximum runtime, stops only its worker process groups, and refuses mid-run code changes. It never starts a training process. Stage receipts and per-image outputs support resume with the same code. When changing algorithms, use a fresh artifact root rather than reusing scored outputs.

The portable code depends on adjacent `flatlay_cutter` and `leffa_native` packages plus the official LeFFA checkout and pinned weights. Set `LEFFA_WORKSPACE` to the local workspace if this folder is moved. Model snapshots and dataset defaults are configured in `common.py`, `data.py`, and `probe.py`.

Important artifacts: `VERIFIED.json` (forward parity), `SMOKE.json` (real GPU check), `SELECTION.json` (frozen readout), `RESULT.json` (held-out gates), `STATE.json`/`WORKER.json` (status), `data/*/IDENTITY.json` (mask provenance), and `site/index.html` (visual evidence).

## Interpretation

The corrected mask can prevent the inpainting input from erasing skin or a skirt. It does not prove the generator preserves those pixels in its raw decoded prediction. The report deliberately shows the raw single-pass image, without compositing the protected original pixels over it. Source routing, causal contribution, and visible semantic detail accuracy are distinct measurements; all three are required before claiming that a mark identifies a generated detail.
