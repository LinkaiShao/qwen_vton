"""Build a static, independently reviewable Molmo point-audit report."""
import argparse,json,shutil,hashlib,csv
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/nvme0/leffa_native/molmo_view_audit_20260929')
VIEWS=['flatlay','worn','generated']
LABELS={'flatlay':'Flatlay','worn':'Real worn photo','generated':'LeFFA try-on'}
COLORS={'flatlay':'#2864d7','worn':'#ac46b3','generated':'#df7509'}

def read(p):return json.loads(p.read_text())
def build(root,out):
 m=read(root/'manifest.json');s=read(root/'summary.json');a=read(root/'adjudicated.json')
 out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True);(out/'data').mkdir(exist_ok=True)
 for row in m['records']:
  for v,src in row['views'].items():
   Image.open(src).convert('RGB').save(out/'images'/f'{row["id"]}_{v}.jpg',quality=93,optimize=True)
 public_manifest={k:v for k,v in m.items() if k!='records'}
 public_manifest['records']=[{'id':r['id'],'hashes':r['hashes'],'images':{v:f'images/{r["id"]}_{v}.jpg' for v in VIEWS}} for r in m['records']]
 public_manifest['image_note']='Browser images are JPEG presentation copies. Coordinates refer to the original full-resolution images; hashes identify those originals.'
 for name,obj in [('manifest.json',public_manifest),('summary.json',s),('adjudicated.json',a),('blind_review.json',read(root/'blind_review.json')),('point_review.json',read(root/'point_review.json')),('BLIND_REVIEW_LOCK.json',read(root/'BLIND_REVIEW_LOCK.json'))]:
  (out/'data'/name).write_text(json.dumps(obj,indent=2)+'\n')
 for filename in ['molmo_view_audit.py','summarize_molmo_audit.py','show_molmo_audit.py','molmo_flatlay.py']:
  shutil.copyfile(Path(__file__).parent/filename,out/'data'/filename)
 with (out/'data'/'checks.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['id','view','concept','visible_reference','minimum_instances','outcome','errors','correct_points','wrong_points','uncertain_points'])
  for x in a:w.writerow([x['id'],x['view'],x['concept'],x['reference']['status'],x['reference']['visible_instances'],x['status'],';'.join(x['errors']),*[sum(p['judgment']==g for p in x['points']) for g in ['correct','wrong','uncertain']]])
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(12.8,4.3),layout='constrained')
 for v in VIEWS:
  ax.plot(range(10),[s['concepts'][v][c]['failure_pct'] for c in m['concepts']],label=LABELS[v],color=COLORS[v],marker='o',linewidth=2,markersize=6)
 ax.set_xticks(range(10),[x.replace(' ','\n').title() for x in m['concepts']]);ax.set_ylim(-3,105);ax.set_yticks(range(0,101,20));ax.set_ylabel('Failed detail checks (%)');ax.grid(axis='y',alpha=.18);ax.legend(ncols=3,loc='upper center',bbox_to_anchor=(.5,1.16),frameon=False)
 fig.savefig(out/'failure_by_detail.svg');fig.savefig(out/'failure_by_detail.png',dpi=170);plt.close(fig)
 data={'manifest':public_manifest,'summary':s,'checks':a,'labels':LABELS}
 (out/'audit-data.js').write_text('window.AUDIT = '+json.dumps(data,separators=(',',':'))+';\n')
 template=Path(__file__).with_name('molmo_audit_template.html').read_text()
 (out/'index.html').write_text(template)
 shutil.copyfile(Path(__file__).with_name('molmo_audit_template.html'),out/'data'/'molmo_audit_template.html')
 print('REPORT',out,'assets',len(list(out.rglob('*'))),'MiB',round(sum(x.stat().st_size for x in out.rglob('*') if x.is_file())/2**20,2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path,default=ROOT/'site');a=p.parse_args();build(a.root,a.out)
