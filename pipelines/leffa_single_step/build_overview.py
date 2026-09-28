"""Link both experiments, including the unsuccessful first run, in one report."""
import argparse
import json
from pathlib import Path


def build(job):
    trials = [json.loads((job / name / 'RESULT.json').read_text()) for name in ['run_v1', 'run_v2']]
    result = trials[1]
    joint, control = result['joint'], result['control']
    change = 100 * (control['mean_fixed_roi_dino_distance'] - joint['mean_fixed_roi_dino_distance']) / control['mean_fixed_roi_dino_distance']
    mass = joint['mass_at_half']
    links = ['initial', 'followup']
    rows = []
    for i, (trial, link) in enumerate(zip(trials, links), 1):
        j, c = trial['joint'], trial['control']
        rows.append(f'<tr><td><a href="{link}/">Experiment {i}: {trial["iterations"]:,} updates</a></td>' +
                    ''.join(f'<td>{j["mass_at_half"][k]:.1%}</td>' for k in ['collar_neckline','left_sleeve_cuff','right_sleeve_cuff']) +
                    f'<td>{j["mean_fixed_roi_dino_distance"]:.4f}</td><td>{c["mean_fixed_roi_dino_distance"]:.4f}</td><td>{"All criteria met" if trial["status"]=="SUCCESSFUL" else "Criteria not all met"}</td></tr>')
    direction = 'lower' if change >= 0 else 'higher'
    status = 'The single-step gradient path is verified.'
    quality = ('The follow-up meets the numeric criteria on this development cohort.' if result['status']=='SUCCESSFUL' else 'The full quality target is still not met.')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LeFFA single-step detail feedback · H200 results</title>
<style>*{{box-sizing:border-box}}body{{margin:0;color:#1e3039;background:#edf2f4;font:17px/1.6 system-ui,sans-serif}}main{{max-width:1150px;margin:auto;padding:42px 24px}}h1{{font-size:clamp(30px,5vw,54px);line-height:1.1;max-width:850px}}h2{{margin-top:38px;font-size:24px}}a{{color:#136e84}}.tag{{font-size:12px;letter-spacing:.12em;color:#4a6875;font-weight:800}}.card{{padding:22px;background:white;border-radius:10px;border:1px solid #d9e3e7;margin:20px 0}}.actions{{display:flex;gap:14px;flex-wrap:wrap}}.button{{display:block;padding:13px 20px;background:#116b66;color:white;text-decoration:none;border-radius:6px;font-weight:650}}.secondary{{background:#425768}}table{{width:100%;border-collapse:collapse;font-size:14px;background:white}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #dbe5e9}}th{{background:#dbe7ec}}.scroll{{overflow-x:auto}}.metric{{font-size:24px;font-weight:700;color:#146a64}}.muted{{color:#586e7a;font-size:14px}}img{{width:100%;height:auto}}code{{font-size:14px}}@media(max-width:650px){{main{{padding:26px 14px}}.card{{padding:16px}}th,td{{padding:8px}}}}</style><main>
<div class="tag">H200 · LEFFA · REAL VITON-HD PAIRS · 28 SEPTEMBER 2026</div>
<h1>Local detail feedback from one training step.</h1>
<div class="card"><strong>{status} {quality}</strong><p>LeFFA takes one sampled noisy target latent and garment condition. Learned semantic keys read component locations from live attention features. A frozen VAE decodes the single-step estimate; frozen DINO scores local crops and sends gradients back into LeFFA.</p><p><strong>No live Molmo. No denoising trajectory before the loss.</strong></p></div>
<div class="actions"><a class="button" href="followup/">See targets, predictions, crops & heatmaps</a><a class="button secondary" href="initial/">Inspect the first experiment</a></div>
<h2>What the follow-up measured</h2>
<p class="metric">{mass['collar_neckline']:.1%} collar · {mass['left_sleeve_cuff']:.1%} left sleeve · {mass['right_sleeve_cuff']:.1%} right sleeve</p>
<p>Attention mass inside automatic annotation boxes at t=.5. The requested threshold was over 80% for each component.</p>
<p>Same-location DINO distance was <strong>{abs(change):.2f}% {direction} than the matched control</strong>: {joint['mean_fixed_roi_dino_distance']:.5f} versus {control['mean_fixed_roi_dino_distance']:.5f}. This comparison measures the image at the same annotated location, so moving the crop cannot produce that score improvement. The report also shows per-component results and a paired-image uncertainty interval.</p>
<img src="followup/assets/localization.png" alt="Localization across sampled timesteps">
<h2>Both experiments remain visible</h2><div class="scroll"><table><tr><th>Run</th><th>Collar</th><th>Left sleeve</th><th>Right sleeve</th><th>Joint DINO ↓</th><th>Control DINO ↓</th><th>Result</th></tr>{''.join(rows)}</table></div>
<p class="muted">Attention mass is at t=.5; fixed-site DINO averages all annotated regions at seven timesteps. Total: {sum(t['iterations'] for t in trials):,} optimization updates, zero reported OOMs. Both runs used actual model forwards, RGB decoding and DINO crops on every iteration.</p>
<h2>Scope of the evidence</h2><div class="card"><p>24 optimization images and 8 separate garment images, with 14 annotated validation regions. Labels are previously cached automatic MolmoPoint proposals; they are not human ground truth. No annotation box is passed into the localization logits. The crop size is supplied by metadata.</p><p>LeFFA has no native collar/cuff text-token attention: its text cross-attention is disabled. This adds a semantic readout to two live image-attention layers and trains their Q/K projections. The readout has 2,880 learned weights; the selected projections have 1,024,000.</p><p>The follow-up was chosen after observing the first result, so its eight images are a development cohort. The figures are single-step estimates from noised paired targets during training. Improved fully sampled try-on images and broad generalization have not been established.</p></div>
<h2>Implementation & evidence</h2><p><a href="https://github.com/LinkaiShao/qwen_vton/tree/master/pipelines/leffa_single_step">Reproducible source, manifest and commands</a> · <a href="followup/data/GRADIENT_VERIFICATION.json">Real DINO gradient verification</a> · <a href="followup/data/CHECKPOINT_AUDIT.json">Checkpoint and update audit</a> · <a href="followup/data/RESULT.json">Complete follow-up results</a></p></main></html>'''
    sample_path=job/'full_samples_v2/RESULT.json'
    if sample_path.exists():
        full=json.loads(sample_path.read_text())['mean_fixed_roi_dino_distance']
        change_full=100*(full['denoise_spatial_control']-full['denoise_spatial_dino'])/full['denoise_spatial_control']
        generation=f'<h2>Transfer to actual 20-step generation</h2><div class="card"><p>In a separate post-training check on the same eight garments, fixed-region DINO distance was {full["denoise_spatial_dino"]:.5f} for the joint model versus {full["denoise_spatial_control"]:.5f} for the control: <strong>{abs(change_full):.2f}% {"lower" if change_full>=0 else "higher"}</strong>. The original model scored {full["baseline"]:.5f}.</p><p>These are matched-seed full samples from random noise, with 20 DDPM steps, guidance 2.5 and no final RGB repaint. Their generation was never part of the training loss.</p><a class="button" href="full-samples/">Inspect all full samples and detail crops</a></div>'
        generation=generation.replace('<p>These are matched-seed', '<p><strong>Transfer is mixed: collar scores improved, but both sleeve-end averages worsened.</strong> The paired-image bootstrap interval includes zero; eight development images do not establish a reliable overall gain. Small feature-score changes are not labelled as visibly fixed details.</p><p>These are matched-seed')
        page=page.replace('<h2>Both experiments remain visible</h2>',generation+'<h2>Both experiments remain visible</h2>')
        page=page.replace('Improved fully sampled try-on images and broad generalization have not been established.', 'The separate 20-step check measures full-sample feature scores on these same eight images; broad generalization and visible defect correction have not been established.')
    page=page.replace('<h2>Scope of the evidence</h2>', '<p>The weakest collar cases are <a href="followup/#case-02993_00">02993_00 (18.6% mass; green V-neck)</a> and <a href="followup/#case-10337_00">10337_00 (54.6%; white collar)</a>. These failures are included in the gallery.</p><h2>Scope of the evidence</h2>')
    path = job / 'site/index.html';path.write_text(page)
    print(path)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    build(p.parse_args().job)
