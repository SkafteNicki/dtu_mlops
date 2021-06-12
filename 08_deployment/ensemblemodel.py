import torch
from torch import nn
from torchvision import models

# TODO: add more
backbones = ['resnet18']

class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbones = nn.ModuleList([getattr(models, bb)(pretrained=True) for bb in backbones])

    def forward(self, x: torch.Tensor):
        res = [bb(x) for bb in self.backbones]
        
        # todo: combine the output in res
        