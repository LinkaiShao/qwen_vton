"""Publish only this audit's code/report under the existing GitHub Pages site."""
import fcntl
import shutil
import subprocess
from common import *
from report import build

REPO=Path('/mnt/nvme0/vton_region_benchmark/20260925_eight_models_v2/publish-github')
LOCK=Path('/mnt/nvme0/sku_partition_veto/20260927_v1/site_publish.lock')
URL='https://linkaishao.github.io/qwen_vton/leffa-native/localization/'

def publish(root,repo=REPO):
    build(root);LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
        staged=git('diff','--cached','--name-only')
        if staged:raise RuntimeError('Publication checkout has staged work; leave it untouched')
        report_path='docs/leffa-native/localization';code_path='pipelines/leffa_localization'
        shutil.copytree(root/'site',repo/report_path,dirs_exist_ok=True)
        (repo/code_path).mkdir(parents=True,exist_ok=True)
        for p in HERE.iterdir():
            if p.suffix in ('.py','.md'):shutil.copy2(p,repo/code_path/p.name)
        git('add','--',report_path,code_path)
        changed=git('diff','--cached','--name-only')
        if changed:
            allowed=(report_path+'/',code_path+'/')
            if any(not p.startswith(allowed) for p in changed.splitlines()):raise RuntimeError('Unexpected staged path')
            git('commit','-m','Publish frozen LeFFA detail localization audit')
        branch=git('branch','--show-current')
        if not branch:raise RuntimeError('Detached publication checkout')
        # No force push, reset, rebase, or modifications to other reports.
        git('push','origin','HEAD:'+branch)
        receipt={'url':URL,'commit':git('rev-parse','HEAD'),'updated':time.time()}
        atomic(root/'PUBLISHED.json',receipt);print('PUBLISHED',URL,receipt['commit'],flush=True)

if __name__=='__main__':
    p=arguments(__doc__);p.add_argument('--repo',type=Path,default=REPO);a=p.parse_args();publish(a.root,a.repo)
