"""Combine frozen blind presence review and explicit point adjudication; never infer correctness from counts."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path('/mnt/nvme0/leffa_native/molmo_view_audit_20260929')

def read(p):return json.loads(p.read_text())
def main(root):
 m=read(root/'manifest.json');b=read(root/'blind_review.json');p=read(root/'point_review.json')
 assert hashlib.sha256((root/'blind_review.json').read_bytes()).hexdigest()==read(root/'BLIND_REVIEW_LOCK.json')['sha256']
 bs={(x['id'],x['view']):x for x in b['records']};ps={(x['id'],x['view']):x for x in p['records']};out=[]
 assert len(bs)==len(ps)==96
 for row in m['records']:
  for view in m['views']:
   key=(row['id'],view);br=bs[key];pr=ps[key];raw=read(root/'predictions'/f'{key[0]}__{view}.json')
   assert raw['complete'] and raw['image_sha256']==row['hashes'][view]
   for pred in raw['records']:
    c=pred['concept'];ref=br['parts'][c];pts=pred['points'];manual=pr['parts'].get(c)
    flags=[x for x in ['invalid_raw_coordinates','token_limit_reached','unparsed_point_tag'] if pred.get(x)]
    if ref['status']=='uncertain':grades=['uncertain']*len(pts);covered=None;status='unresolved';errors=[]
    elif ref['status']=='absent':
     grades=['wrong']*len(pts);covered=0;status='fail' if pts or flags else 'pass';errors=['false_positive'] if pts else []
    else:
     assert manual is not None or not pts,(key,c,len(pts),'Missing explicit point review')
     grades=manual['point_judgments'] if manual else [];covered=manual['distinct_visible_instances_covered'] if manual else 0
     assert len(grades)==len(pts),(key,c,len(grades),len(pts))
     assert covered<=grades.count('correct'),(key,c,'Impossible coverage')
     minimum=ref['visible_instances'];unknown=grades.count('uncertain');errors=[]
     if 'wrong' in grades:errors.append('mislocalized')
     if covered==0 and unknown==0:errors.append('missed')
     elif covered+unknown<minimum:errors.append('partial_coverage')
     status='fail' if errors else ('unresolved' if unknown or covered<minimum else 'pass')
    if flags:
     errors.append('protocol');status='fail' if ref['status']!='uncertain' else 'unresolved'
    out.append({'id':key[0],'view':view,'concept':c,'reference':ref,'status':status,'errors':errors,
      'points':[{ 'xy':xy,'judgment':grade} for xy,grade in zip(pts,grades)],'covered':covered,
      'raw':pred['raw'],'protocol_flags':flags,'reference_note':br['note'],'review_note':pr['note']})
 assert len(out)==960
 def summarize(rows):
  statuses=Counter(x['status'] for x in rows);grades=Counter(y['judgment'] for x in rows for y in x['points']);errors=Counter(e for x in rows for e in x['errors'])
  known=statuses['pass']+statuses['fail'];points_known=grades['correct']+grades['wrong']
  return {'checks':len(rows),'assessable':known,'failed':statuses['fail'],'passed':statuses['pass'],'unresolved':statuses['unresolved'],
    'failure_pct':100*statuses['fail']/known if known else None,'correct_points':grades['correct'],'wrong_points':grades['wrong'],
    'uncertain_points':grades['uncertain'],'wrong_point_pct':100*grades['wrong']/points_known if points_known else None,
    'error_counts':dict(errors),'present':sum(x['reference']['status']=='present' for x in rows),
    'absent':sum(x['reference']['status']=='absent' for x in rows),'reference_uncertain':sum(x['reference']['status']=='uncertain' for x in rows)}
 summary={'all':summarize(out),'views':{},'concepts':{},'inference':read(root/'INFERENCE_COMPLETE.json')}
 for view in m['views']:
  rows=[x for x in out if x['view']==view];s=summarize(rows)
  s['images_with_known_failure']=len({x['id'] for x in rows if x['status']=='fail'})
  for presence in ['present','absent']:s[presence+'_only']=summarize([x for x in rows if x['reference']['status']==presence])
  summary['views'][view]=s
  summary['concepts'][view]={c:summarize([x for x in rows if x['concept']==c]) for c in m['concepts']}
 # Paired cluster bootstrap: resample garment IDs, retaining every concept and view together.
 ids=[x['id'] for x in m['records']];rng=np.random.default_rng(20260929);draw=rng.integers(0,len(ids),size=(10000,len(ids)))
 boot={}
 for view in m['views']:
  f=np.array([sum(x['status']=='fail' for x in out if x['view']==view and x['id']==iid) for iid in ids])
  n=np.array([sum(x['status']!='unresolved' for x in out if x['view']==view and x['id']==iid) for iid in ids])
  boot[view]=100*f[draw].sum(1)/n[draw].sum(1)
  summary['views'][view]['cluster_bootstrap_95pct_ci']=list(np.quantile(boot[view],[.025,.975]))
 summary['paired_differences']={}
 for a,bb in [('worn','flatlay'),('generated','flatlay'),('generated','worn')]:
  summary['paired_differences'][a+' minus '+bb]={'percentage_points':summary['views'][a]['failure_pct']-summary['views'][bb]['failure_pct'],
   'cluster_bootstrap_95pct_ci':list(np.quantile(boot[a]-boot[bb],[.025,.975]))}
 (root/'adjudicated.json').write_text(json.dumps(out,indent=2)+'\n');(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 for v,s in summary['views'].items():
  print(v,'failed',s['failed'],'/',s['assessable'],round(s['failure_pct'],1),'unresolved',s['unresolved'],'wrong points',s['wrong_points'],'/',s['wrong_points']+s['correct_points'],'present-only',s['present_only']['failed'],'/',s['present_only']['assessable'],'absent-only',s['absent_only']['failed'],'/',s['absent_only']['assessable'])
 print('CONCEPT FAILURES')
 for c in m['concepts']:print(c,[(v,f"{summary['concepts'][v][c]['failed']}/{summary['concepts'][v][c]['assessable']}") for v in m['views']])
 print('DIFFERENCES',summary['paired_differences'])
 print('Protocol problems',[x for x in out if x['protocol_flags']])

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=ROOT);main(a.parse_args().root)
