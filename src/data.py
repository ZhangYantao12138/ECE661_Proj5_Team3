import hashlib,pickle,tarfile,urllib.request
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import save_json,stable_seed
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
URL='https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
MD5='c58f30108f718f92721af3b95e74349a'
def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);arc=root/'cifar-10-python.tar.gz'
    if not arc.exists() or hashlib.md5(arc.read_bytes()).hexdigest()!=MD5:
        partial=arc.with_suffix('.partial')
        # Never replace a verified archive with a partial network response.
        for attempt in range(3):
            try:
                request=urllib.request.Request(URL,headers={'User-Agent':'ECE661-data-preparation/1.0'})
                with urllib.request.urlopen(request,timeout=60) as response,partial.open('wb') as output:
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        output.write(chunk)
                if hashlib.md5(partial.read_bytes()).hexdigest()!=MD5:
                    raise ValueError('Incomplete CIFAR download or checksum mismatch')
                break
            except (OSError,ValueError) as exc:
                if attempt==2:
                    raise RuntimeError('CIFAR download failed after 3 attempts. Place the official verified archive in '+str(arc)) from exc
                print('CIFAR download retry',attempt+1,flush=True)
        partial.replace(arc)
    if hashlib.md5(arc.read_bytes()).hexdigest()!=MD5:raise ValueError('CIFAR archive checksum mismatch')
    with tarfile.open(arc) as tf:tf.extractall(root,filter='data')
    return load_arrays(root)
def load_arrays(root):
    root=Path(root)/'cifar-10-batches-py'
    def read(names):
        records=[]
        for name in names:
            with open(root/name,'rb') as f:records.append(pickle.load(f,encoding='bytes'))
        x=np.concatenate([r[b'data'] for r in records]).reshape(-1,3,32,32)
        y=np.concatenate([r[b'labels'] for r in records]).astype('int64');return x,y
    return (*read([f'data_batch_{i}' for i in range(1,6)]),*read(['test_batch']))
def make_splits(labels,seed=661,val_per_class=500,select_per_class=100):
    rng=np.random.Generator(np.random.PCG64(seed));rsel=np.random.Generator(np.random.PCG64(662))
    train=[];val=[];select=[]
    for c in range(10):
        ids=np.flatnonzero(labels==c);ids=rng.permutation(ids)
        v=ids[:val_per_class];val.extend(v.tolist());train.extend(ids[val_per_class:].tolist())
        select.extend(rsel.permutation(np.sort(v))[:select_per_class].tolist())
    out=dict(train=train,val=val,selection=select,seed=seed,classes=CLASSES)
    assert not set(train)&set(val)
    assert set(train)|set(val)==set(range(len(labels)))
    return out
def normalization(images,ids):
    total=np.zeros(3);sq=np.zeros(3);count=0
    for chunk in np.array_split(np.asarray(ids),max(1,len(ids)//256)):
        x=images[chunk].astype('float64')/255
        total+=x.sum((0,2,3));sq+=(x*x).sum((0,2,3));count+=len(x)*1024
    mean=total/count;std=np.sqrt(np.maximum(sq/count-mean**2,0))
    return mean.tolist(),std.tolist()
class RawDataset(Dataset):
    def __init__(self,images,labels,ids=None,split='train',augment=False):
        self.images=images;self.labels=labels;self.ids=np.arange(len(images)) if ids is None else np.asarray(ids)
        self.split=split;self.augment=augment;self._device_cache={}
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        j=int(self.ids[i]);x=torch.from_numpy(self.images[j].copy()).float()/255
        if self.augment:
            x=torch.nn.functional.pad(x,(4,4,4,4));top,left=torch.randint(9,(2,)).tolist();x=x[:,top:top+32,left:left+32]
            if torch.rand(())<.5:x=x.flip(-1)
        return x,int(self.labels[j]),f'{self.split}:{j}'
    def cached_batch(self,positions,device):
        if self.augment:raise ValueError('Augmented supervised data must not be cached')
        key=str(device)
        if key not in self._device_cache:
            # Exactly the original CPU float32 conversion, then one device copy.
            raw=torch.from_numpy(self.images[self.ids].copy()).float()/255
            labels=torch.from_numpy(self.labels[self.ids].copy()).long()
            names=[f'{self.split}:{int(i)}' for i in self.ids]
            self._device_cache[key]=(raw.to(device),labels.to(device),names)
        raw,labels,names=self._device_cache[key]
        positions=np.asarray(positions,dtype=np.int64)
        index=torch.as_tensor(positions,device=device)
        return raw.index_select(0,index),labels.index_select(0,index),[names[int(i)] for i in positions]
    def sequential_batches(self,batch_size,device):
        for start in range(0,len(self),batch_size):
            x,y,names=self.cached_batch(np.arange(start,min(start+batch_size,len(self))),device)
            yield x,y.cpu(),names
def batch_at(dataset,seed,step,batch_size,device):
    rng=np.random.default_rng(stable_seed('train_order',seed,step));ids=rng.choice(len(dataset),batch_size,replace=len(dataset)<batch_size)
    if isinstance(dataset,RawDataset) and not dataset.augment:return dataset.cached_batch(ids,device)
    rows=[dataset[int(i)] for i in ids]
    return torch.stack([r[0] for r in rows]).to(device),torch.tensor([r[1] for r in rows],device=device),[r[2] for r in rows]
