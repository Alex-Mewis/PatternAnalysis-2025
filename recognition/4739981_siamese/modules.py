"""
contains the components of the siamese model.
"""
import torch
from torch import nn
from torchvision import models

class SiameseNetwork(nn.Module):

    def __init__(self) -> None:
        super(SiameseNetwork, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.name = 'siamese'

        return None
    
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        out  = self._backbone(x)
        return out.view(out.size(0), -1)

    def forward(self, xs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        """
        return [self.forward_once(x) for x in xs]

class Classifier(nn.Module):

    def __init__(self, input_dim: int = 512) -> None:
        super(Classifier, self).__init__()

        self._fcl = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.name = 'classifier'

        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._fcl(x)
        return torch.flatten(out)