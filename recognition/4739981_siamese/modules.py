"""
contains the components of the siamese model.
"""
import torch
from torch import nn
from torchvision import models

class SiameseNetwork(nn.Module):

    def __init__(self) -> None:
        super(SiameseNetwork, self).__init__()

        self._backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        return None
    
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        """
        y1 = self._backbone(x1)
        y2 = self._backbone(x2)
        return y1, y2  