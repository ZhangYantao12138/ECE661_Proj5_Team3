import torch
from torch import nn
from .models import build_model
from .utils import load_torch,digest
class ModelBundle(nn.Module):
    def __init__(self,network,mean,std,name,checkpoint_hash='untrained'):
        super().__init__();self.network=network;self.name=name;self.checkpoint_hash=checkpoint_hash
        self.register_buffer('mean',torch.tensor(mean).view(1,3,1,1));self.register_buffer('std',torch.tensor(std).view(1,3,1,1))
    def forward(self,x):return self.network((x-self.mean)/self.std)
    predict_logits=forward
    def freeze(self):
        self.eval()
        for p in self.parameters():p.requires_grad_(False);p.grad=None
        return self
def load_model_bundle(path,device='cpu'):
    obj=load_torch(path);net=build_model(obj['architecture_name']);net.load_state_dict(obj['state_dict'])
    return ModelBundle(net,obj['normalization_mean'],obj['normalization_std'],obj['architecture_name'],digest(path)).to(device).freeze()
