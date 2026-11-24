from pathlib import Path
import json,hashlib,os,time
import numpy as np
import torch
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def atomic(path,writer):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:writer(f);f.flush();os.fsync(f.fileno())
    until=time.monotonic()+5
    while True:
        try:os.replace(tmp,p);return
        except OSError as e:
            if getattr(e,'winerror',0) not in [5,32,33] or time.monotonic()>=until:raise
            time.sleep(.1)
def write_json(p,v):atomic(p,lambda f:f.write((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')))
def read_json(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save_torch(p,v):atomic(p,lambda f:torch.save(v,f))
def save_npz(p,**v):atomic(p,lambda f:np.savez_compressed(f,**v))
def remove_state(root,name):
    root=Path(root).resolve();p=(root/'state'/name).resolve()
    assert p.parent==root/'state' and name in ['current.pt','selector.pt']
    if p.exists():p.unlink()
