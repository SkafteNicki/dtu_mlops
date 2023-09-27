import torch
from torch import nn
from torchvision import models

# TODO: add more
backbones = ["resnet18"]


class EnsembleModel(nn.Module):
    """Create an ensemble model."""

    def __init__(self):
        super().__init__()
        self.backbones = nn.ModuleList([getattr(models, bb)(pretrained=True) for bb in backbones])

    def forward(self, x: torch.Tensor):
        """Forward pass of the network."""
        res = [bb(x) for bb in self.backbones]

        # todo: combine the output in res
