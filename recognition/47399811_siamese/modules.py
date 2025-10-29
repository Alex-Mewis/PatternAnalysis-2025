"""
contains the components of the siamese model.
"""
import os
from datetime import datetime
import torch
from torch import nn
from torchvision import models

#### PERAMBLE #####################################################################
this_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(this_dir, "models")
siamese_models_dir = os.path.join(models_dir, 'siamese')
classifier_models_dir = os.path.join(models_dir, 'classifier')
both_models_dir = os.path.join(models_dir, 'both')

if not os.path.exists(models_dir): os.mkdir(models_dir)
if not os.path.exists(siamese_models_dir): os.mkdir(siamese_models_dir)
if not os.path.exists(classifier_models_dir): os.mkdir(classifier_models_dir)
if not os.path.exists(both_models_dir): os.mkdir(both_models_dir)

#### NETWORKS #####################################################################
class SiameseNetwork(nn.Module):

    def __init__(self) -> None:
        super(SiameseNetwork, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._backbone = nn.Sequential(*list(resnet.children())[:-1])

        self._classififer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 2),
        )
        
        self.name = 'both'

        return None
    
    @property
    def final_convolution_layer(self) -> nn.Module:
        return list(self._backbone.children())[-3] # skip the AvgPool2d & batchNorm2d layers
    
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        out  = self._backbone(x)
        return out.view(out.size(0), -1)
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        latent_vector = self._backbone(x)
        latent_vector = latent_vector.view(latent_vector.size(0), -1)
        return self._classififer(latent_vector)

    def forward(self, xs: list[torch.Tensor] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        """
        if isinstance(xs, list):
            return [self.forward_once(x) for x in xs]
        else:
            return self.forward_once(xs)

class Classifier(nn.Module):

    def __init__(self, input_dim: int = 512) -> None:
        super(Classifier, self).__init__()

        self._fcl = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            # nn.BatchNorm1d(input_dim),
            # nn.Linear(input_dim, 256),
            # nn.GELU(),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(256, 64),
            # nn.GELU(),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(64, 2),

            # nn.Linear(input_dim, 2),
            
            # nn.Linear(input_dim, input_dim),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            
            # nn.Linear(input_dim, input_dim),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.2),

            # nn.Linear(input_dim, input_dim),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            
            # nn.Linear(input_dim, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            
            # nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.2),

            # nn.Linear(128, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            
            # nn.Linear(64, 2),

        )
        
        self.name = 'classifier'

        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fcl(x)


#### FUNCTIONS ####################################################################
def save_model(model: nn.Module, outdir: str | None = None) -> None:
    if outdir is None: outdir = os.path.join(models_dir, model.name)
    filename = f"{model.name}_{datetime.now().timestamp()}.params"
    outpath = os.path.join(outdir, filename)
    torch.save(model.state_dict(), outpath)
    print(f"Saved: {outpath}")
    return None

def load_model(model_name: str, model_path: str | None = None) -> nn.Module:
    
    if model_path is not None:
        model = os.path.basename(model_path).split('_')[0]
    
    if model_name.lower() == 'siamese' or model_name.lower() == 'both':
        model = SiameseNetwork()
    elif model_name.lower() == 'classifier':
        model = Classifier()
    else:
        raise ValueError(f"model must be either 'siamese' or 'classifier' not {model.lower()}")

    if model_path is not None:
        print(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict) 
        return model
    
    latest_model_datetime = None
    latest_model_path = None
    model_state_dir = os.path.join(models_dir, model_name)
    for model_filename in os.listdir(model_state_dir):
        model_timestamp = float(model_filename.replace('.params', '').split('_')[1])
        model_datetime = datetime.fromtimestamp(model_timestamp)
        if latest_model_path is None or model_datetime > latest_model_datetime:
            latest_model_datetime = model_datetime
            latest_model_path = os.path.join(model_state_dir, model_filename)

    if latest_model_path is None:
        print(f"Warning there were no previous models which could be loaded for {model_name}.")
        print("So just staring from a new model.")
        return model

    print(f"Loading model from: {latest_model_path}") 
    state_dict = torch.load(latest_model_path)
    model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    siamese = SiameseNetwork()
    print(siamese)
    print('\n\n')
    classifier = Classifier()
    print(classifier)
