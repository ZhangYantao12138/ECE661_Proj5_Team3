import torch
from torch import nn
import torch.nn.functional as F
class BasicBlock(nn.Module):
    def __init__(self,ci,co,stride=1):
        super().__init__();self.stride=stride;self.co=co
        self.c1=nn.Conv2d(ci,co,3,stride,1,bias=False);self.b1=nn.BatchNorm2d(co)
        self.c2=nn.Conv2d(co,co,3,1,1,bias=False);self.b2=nn.BatchNorm2d(co)
    def forward(self,x):
        shortcut=x[:,:,::self.stride,::self.stride]
        extra=self.co-x.shape[1]
        if extra:shortcut=F.pad(shortcut,(0,0,0,0,extra//2,extra-extra//2))
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))+shortcut)
class ResNet20(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Sequential(nn.Conv2d(3,16,3,1,1,bias=False),nn.BatchNorm2d(16),nn.ReLU())
        blocks=[];ci=16
        for co in (16,32,64):
            for j in range(3):blocks.append(BasicBlock(ci,co,2 if j==0 and ci!=co else 1));ci=co
        self.blocks=nn.Sequential(*blocks);self.fc=nn.Linear(64,10)
    def forward(self,x):return self.fc(self.blocks(self.stem(x)).mean((2,3)))
class VGG11(nn.Module):
    def __init__(self):
        super().__init__();layers=[];ci=3
        for co in [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M']:
            if co=='M':layers.append(nn.MaxPool2d(2))
            else:layers.extend([nn.Conv2d(ci,co,3,padding=1),nn.BatchNorm2d(co),nn.ReLU()]);ci=co
        self.features=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(512,512),nn.ReLU(),nn.Dropout(.5),nn.Linear(512,512),nn.ReLU(),nn.Dropout(.5),nn.Linear(512,10))
    def forward(self,x):return self.head(self.features(x))
class DenseLayer(nn.Module):
    def __init__(self,ci):
        super().__init__();self.net=nn.Sequential(nn.BatchNorm2d(ci),nn.ReLU(),nn.Conv2d(ci,48,1,bias=False),nn.BatchNorm2d(48),nn.ReLU(),nn.Conv2d(48,12,3,padding=1,bias=False))
    def forward(self,x):return torch.cat((x,self.net(x)),1)
class DenseNet40(nn.Module):
    def __init__(self):
        super().__init__();layers=[nn.Conv2d(3,24,3,padding=1,bias=False)];ci=24
        for b in range(3):
            for _ in range(6):layers.append(DenseLayer(ci));ci+=12
            if b<2:
                co=ci//2;layers.extend([nn.BatchNorm2d(ci),nn.ReLU(),nn.Conv2d(ci,co,1,bias=False),nn.AvgPool2d(2)]);ci=co
        assert ci==132
        layers.extend([nn.BatchNorm2d(ci),nn.ReLU()]);self.features=nn.Sequential(*layers);self.fc=nn.Linear(ci,10)
    def forward(self,x):return self.fc(self.features(x).mean((2,3)))
FACTORY={'resnet20':ResNet20,'vgg11_bn_cifar':VGG11,'densenet_bc40_k12':DenseNet40}
def build_model(name):return FACTORY[name]()
