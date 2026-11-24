import hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import torch

def stable_seed(*fields):
    # Canonical UTF-8 JSON, first eight SHA256 bytes, little endian.
    b=json.dumps(fields,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(b).digest()[:8],'little')
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)
def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    elif str(device).startswith('mps'):torch.mps.synchronize()
def environment():
    return dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
def save_torch(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');torch.save(obj,tmp);tmp.replace(p)
    save_json(str(p)+'.sha256.json',{'sha256':digest(p)})
def load_torch(path,device='cpu'):
    return torch.load(path,map_location=device,weights_only=False)
