"""
contains the components of the siamese model.
Also has helpful functions for saving and loaded the model.

Made by: Alexander Mewis
"""
import os
from datetime import datetime
import torch
from torch import nn
from torchvision import models

#### PERAMBLE #####################################################################
models_dir = os.path.join(os.getcwd(), "models")
if not os.path.exists(models_dir): os.mkdir(models_dir)

#### NETWORKS #####################################################################
class SiameseNetwork(nn.Module):
    """
    Contains both the CNN backbone and the binary classifier.
    
    CNN backbone: modified resnet18.
    binary classifier: some fully connected layers.
    """
    def __init__(self) -> None:
        """
        Initialise all modules used.
        """
        super(SiameseNetwork, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._backbone = nn.Sequential(*list(resnet.children())[:-1])

        self._classififer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(64, 2),
        )
        
        return None
    
    @property
    def final_convolution_layer(self) -> nn.Module:
        """ returns the final convolution layer in the CNN backbone. """
        return list(self._backbone.children())[-3] # skip the AvgPool2d & batchNorm2d layers
    
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Forward the given x (images) through the CNN backbone.

        Parameters:
            x [torch.Tensor]: a tensor of images in the shape:
                (batch_size, num_channels, height, width).

        Returns:
            [torch.Tensor] a 512 dimensional latent vector containing
                the images features. Output shape is: (batch_size, 512). 
        """
        out  = self._backbone(x)
        return out.view(out.size(0), -1)
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify the images (x) by feeding in the output of the CNN into
        the binary classifier.

        Parameters:
            x [torch.Tensor]: a tensor of images in the shape:
                (batch_size, num_channels, height, width).

        Returns:
            [torch.Tensor]: the models confidence of each class across the batches
                Output shape is: (batch_size, 2). 
        """
        latent_vector = self.forward_once(x)
        return self._classififer(latent_vector)

    def forward(self, xs: list[torch.Tensor] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forwards either a list of tensors or just a single tensor.

        If xs is a list of tensors then returns the element-wise forward of each tensor.
        Otherwise, just perform a normal forward on the signal tensor. 
        """
        if isinstance(xs, list):
            return [self.forward_once(x) for x in xs]
        else:
            return self.forward_once(xs)



#### FUNCTIONS ####################################################################
def save_model(model: nn.Module) -> None:
    """
    Saves the models state_dict to the models_dir directory.

    Parameters:
        model [nn.Module]: the model to be saved.
    """
    filename = f"model_{datetime.now().timestamp()}.params"
    outpath = os.path.join(models_dir, filename)
    torch.save(model.state_dict(), outpath)
    print(f"Saved: {outpath}")
    return None

def load_model(model_path: str | None = None) -> nn.Module:
    """
    loads the model from the previously saved models.
    
    Parameters:
        model_path [str]: if specified then the models state_dict
            is taken from the given filepath otherwise the most recently
            saved model is taken.

    Returns:
        [nn.Module]: the model which has been loaded.
    """

    model = SiameseNetwork()

    if model_path is not None:
        print(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict) 
        return model

    # otherwise we fetch the most recent model file in the models_dir. 
    latest_model_datetime = None
    latest_model_path = None
    for model_filename in os.listdir(models_dir):
        if not model_filename.endswith('.params'): continue

        model_timestamp = float(model_filename.replace('.params', '').split('_')[1])
        model_datetime = datetime.fromtimestamp(model_timestamp)
        if latest_model_path is None or model_datetime > latest_model_datetime:
            latest_model_datetime = model_datetime
            latest_model_path = os.path.join(models_dir, model_filename)

    if latest_model_path is None:
        # then no models have been saved to the models_dir
        print(f"Warning there were no previous models which could be loaded.")
        print("So just staring from a new model.")
        return model

    print(f"Loading model from: {latest_model_path}") 
    state_dict = torch.load(latest_model_path) 
    model.load_state_dict(state_dict)

    return model
    

if __name__ == "__main__":
    siamese = SiameseNetwork()
    print(siamese)
