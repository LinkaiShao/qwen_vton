"""Strict localization gate. No DINO fine-tuning is launched by this audit."""
import numpy as np
from common import *
from metrics import score_map,bootstrap

def evaluate(root):
    selection=read(root/'SELECTION.json');rows=[r for r in manifest(root) if r['role']=='heldout']
    observations=[];causal=[];mask_ready=0;annotation_uncertain={p:0 for p in PARTS}
    for info in rows:
        f=root/'data'/info['key'];meta=read(f/'IDENTITY.json')
        if meta['status']=='ready':mask_ready+=1
        targetpath=root/'annotations/target'/(info['key']+'.json')
        target=read(targetpath) if targetpath.exists() else {'parts':{p:{'status':'uncertain'} for p in PARTS}}
        for p in PARTS:
            if target['parts'][p]['status']=='uncertain':annotation_uncertain[p]+=1
        for t in TIMESTEPS:
            path=root/'probe'/info['key']/f'{t}.json';probe=read(path) if path.exists() else None
            genpath=root/'annotations/generated'/info['key']/f'{t}.json';gen=read(genpath) if genpath.exists() else None
            gm=np.load(genpath.with_suffix('.npz'))['masks'] if gen else None
            maps=np.load(path.with_suffix('.npz'))['maps'] if probe else None
            for j,p in enumerate(PARTS):
                intended_visible=target['parts'][p]['status']=='present'
                s=probe['scores'][p] if probe else {}
                actual_visible=bool(gen and gen['parts'][p]['status']=='present')
                actual=score_map(maps[j],gm[j]) if probe and actual_visible else None
                observations.append({'key':info['key'],'part':p,'t':t,'bin':t//200,
                    'intended_visible':intended_visible,'actual_visible':actual_visible,
                    'confident':bool(s.get('confident',False)),'intended':s.get('intended'),
                    'actual':actual,'generated_annotation_status':gen['parts'][p]['status'] if gen else 'unavailable'})
            cp=root/'causal'/info['key']/f'{t}.json'
            if cp.exists():
                c=read(cp)
                for p,data in c['parts'].items():
                    interventions=data['interventions'];remove=interventions.get('remove',{});increase=interventions.get('increase',{});unrelated=interventions.get('unrelated',{})
                    causal.append({'key':info['key'],'part':p,'bin':t//200,
                        'remove_delta':remove.get('delta',0.) if remove.get('signal_above_repeat') else 0.,
                        'increase_delta':increase.get('delta',0.) if increase.get('signal_above_repeat') else 0.,
                        'source_specific_delta':remove.get('localized_fraction',0.)-unrelated.get('localized_fraction',0.) if remove.get('signal_above_repeat') and data.get('equal_area_source_control') and unrelated else 0.,
                        'signal':bool(remove.get('signal_above_repeat',False))})
    summary={}
    for p in PARTS:
        bybin={};visible_keys={r['key'] for r in observations if r['part']==p and r['intended_visible']}
        for b in range(5):
            group=[r for r in observations if r['part']==p and r['bin']==b and r['intended_visible']]
            accepted=[r for r in group if r['confident'] and r['intended'] is not None]
            actual=[r for r in accepted if r['actual'] is not None]
            cs=[r for r in causal if r['part']==p and r['bin']==b]
            metrics={'expected_visible':len(group),'confident':len(accepted),'coverage':len(accepted)/max(1,len(group)),
                     'peak_hit':float(np.mean([r['intended']['peak_hit'] for r in accepted])) if accepted else 0.,
                     'inside_mass':float(np.mean([r['intended']['inside_mass'] for r in accepted])) if accepted else 0.,
                     'actual_coverage':len(actual)/max(1,len(group)),
                     'actual_peak_hit':float(np.mean([r['actual']['peak_hit'] for r in actual])) if actual else 0.,
                     'actual_inside_mass':float(np.mean([r['actual']['inside_mass'] for r in actual])) if actual else 0.,
                     'causal_count':len(cs),'causal_signal_fraction':float(np.mean([r['signal'] for r in cs])) if cs else 0.,
                     'causal':{k:bootstrap([r[k] for r in cs]) for k in ['remove_delta','increase_delta','source_specific_delta']}}
            metrics['passed']=bool(len(visible_keys)>=30 and metrics['coverage']>=.8 and metrics['peak_hit']>=.9 and metrics['inside_mass']>=.8
                    and metrics['actual_coverage']>=.8 and metrics['actual_peak_hit']>=.9 and metrics['actual_inside_mass']>=.8
                    and metrics['causal_signal_fraction']>=.8 and all(s and s['ci95'][0]>0 for s in metrics['causal'].values()))
            bybin[str(b)]=metrics
        summary[p]={'visible_images':len(visible_keys),'uncertain_target_annotations':annotation_uncertain[p],'by_noise_bin':bybin,
                    'passed':all(v['passed'] for v in bybin.values())}
    verified=read(root/'VERIFIED.json');sweep=read(root/'SWEEP.json')
    pass_all=all(s['passed'] for s in summary.values()) and verified['passed'] and sweep['finite'] and mask_ready/len(rows)>=.8
    result={'status':'LOCALIZATION_GATE_PASSED' if pass_all else 'LOCALIZATION_GATE_FAILED',
            'training_enabled':False,'dino_training_authorized_by_audit':False,
            'next_action':'Review this localization evidence before designing appearance training.' if pass_all else 'Stop here. Localize the failing components/noise ranges before any appearance training.',
            'heldout_images':len(rows),'corrected_masks_ready':mask_ready,'components':summary,
            'annotations':'Frozen SAM3 two-prompt consensus masks; Qwen only names the source garment. Intended and generated-image masks are imperfect automatic evaluation proxies, not human ground truth. Undetected details count as uncertain, not absent.',
            'selection':selection,'readout_verification':verified,'sweep':sweep}
    atomic(root/'OBSERVATIONS.json',{'records':observations,'causal':causal});atomic(root/'RESULT.json',result)
    print('LOCALIZATION_RESULT',result['status'],flush=True)

if __name__=='__main__':a=arguments(__doc__).parse_args();evaluate(a.root)
