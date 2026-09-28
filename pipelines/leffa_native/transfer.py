"""Authenticated, read-only, single-archive transfer when Contents API is slow.

This workspace utility exposes only one fixed task archive behind a random
Bearer credential. No directory listings or arbitrary filesystem paths exist.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import tarfile
import time
import shutil


def pack(root):
    transport=root/'transport';transport.mkdir(exist_ok=True)
    secret=transport/'secret'
    if not secret.exists():secret.write_text(secrets.token_urlsafe(48));secret.chmod(0o600)
    dest=transport/'dataset.tar'
    with tarfile.open(dest,mode='w') as tar:
        tar.add(root/'data',arcname='data')
        tar.add(root/'manifest.json',arcname='manifest.json')
    h=hashlib.sha256()
    with dest.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    (transport/'archive.json').write_text(json.dumps({'bytes':dest.stat().st_size,'sha256':h.hexdigest()}))
    print('PACKED',dest.stat().st_size,flush=True)


def serve(root):
    archive=root/'transport/dataset.tar';token=(root/'transport/secret').read_text()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/dataset.tar' or not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.send_error(404);return
            size=archive.stat().st_size;start=0
            rg=self.headers.get('Range','')
            if rg:
                try:
                    if not rg.startswith('bytes=') or not rg.endswith('-'):raise ValueError()
                    start=int(rg[6:-1])
                    if not 0<=start<size:raise ValueError()
                except ValueError:self.send_error(416);return
            self.send_response(206 if start else 200)
            self.send_header('Content-Type','application/octet-stream')
            self.send_header('Content-Length',str(size-start));self.send_header('Accept-Ranges','bytes')
            if start:self.send_header('Content-Range',f'bytes {start}-{size-1}/{size}')
            self.end_headers()
            try:
                with archive.open('rb') as f:
                    f.seek(start)
                    for block in iter(lambda:f.read(1<<20),b''):self.wfile.write(block)
            except (BrokenPipeError,ConnectionResetError):pass
        def log_message(self,*_):pass
    print('ARCHIVE_SERVER_READY',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()


def receive(root):
    import requests
    config=json.loads((root/'transport.json').read_text())
    dest=root/'dataset.tar';total=config['bytes'];start=time.time()
    for attempt in range(20):
        offset=dest.stat().st_size if dest.exists() else 0
        if offset==total:break
        token=(root/'.transport_secret').read_text()
        headers={'Authorization':'Bearer '+token}
        if offset:headers['Range']=f'bytes={offset}-'
        try:
            with requests.get(config['url']+'/dataset.tar',headers=headers,stream=True,timeout=(30,90)) as r:
                r.raise_for_status()
                if offset and r.status_code!=206:raise RuntimeError('Server did not honor resume')
                with dest.open('ab' if offset else 'wb') as f:
                    for chunk in r.iter_content(1<<20):
                        f.write(chunk)
                        if f.tell()//(64<<20)!=(f.tell()-len(chunk))//(64<<20):
                            print('RECEIVED_MIB',f.tell()//(1<<20),'SECONDS',round(time.time()-start,1),flush=True)
        except requests.RequestException:
            time.sleep(3)
    if dest.stat().st_size!=total:raise RuntimeError('Incomplete archive transfer')
    h=hashlib.sha256()
    with dest.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    if h.hexdigest()!=config['sha256']:raise RuntimeError('Archive checksum mismatch')
    # Avoid tens of thousands of synchronous small writes to Harbor NFS. The
    # authenticated, checksummed archive stays persistent for future restores.
    staging=Path('/tmp')/('leffa_native_'+config['sha256'][:16]);staging.mkdir(exist_ok=True)
    (root/'DATA_READY.json').unlink(missing_ok=True)
    with tarfile.open(dest) as tar:tar.extractall(staging,filter='data')
    target=root/'data'
    if target.is_symlink():target.unlink()
    elif target.exists():target.rename(root/('data_partial_'+str(int(time.time()))))
    target.symlink_to(staging/'data',target_is_directory=True)
    shutil.copy2(staging/'manifest.json',root/'manifest.json')
    manifest=json.loads((root/'manifest.json').read_text())
    assert all((root/'data'/r['key']/'READY.json').exists() for r in manifest['records'])
    (root/'DATA_READY.json').write_text(json.dumps({'archive_sha256':config['sha256'],'counts':manifest['counts'],
                                                 'transport_seconds':time.time()-start,'data_location':str(staging/'data')}))
    (root/'.transport_secret').unlink(missing_ok=True)
    print('TRANSFER_COMPLETE',manifest['counts'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pack','serve','receive']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.root)
