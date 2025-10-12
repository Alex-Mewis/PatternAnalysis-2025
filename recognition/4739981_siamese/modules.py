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
    
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        """
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        return y1, y2 


class Classifier(nn.Module):

    def __init__(self, input_dim: int = 1000) -> None:
        super(Classifier, self).__init__()

        self._fcl = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._fcl(x)
        return torch.flatten(out)