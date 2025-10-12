"""
contains code for traning, validating, testing and saving the model.
"""
import os, time
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modules import SiameseNetwork

#### PERAMBLE #####################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
this_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(this_dir, "models")
if not os.path.exists(models_dir): os.mkdir(models_dir)


#### HYPERPARAMETERS ##############################################################
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2 

#### LOSS #########################################################################
# TAKEN FROM: https://medium.com/analytics-vidhya/a-friendly-introduction-to-siamese-networks-283f31bf38cd
class ContrastiveLoss(nn.Module):
   
    def __init__(self, margin: float = 2.0) -> None:
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        return None

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: int) -> float:
        # Find the pairwise distance or eucledian distance of two output feature vectors
        euclidean_distance = F.pairwise_distance(output1, output2)
        # perform contrastive loss calculation with the distance
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

#### MODEL FUNCTIONS #############################################################
def save_model(model: SiameseNetwork, outdir: str | None = None) -> None:
    if outdir is None: outdir = models_dir
    filename = f"siamese_{datetime.now().timestamp()}.params"
    outpath = os.path.join(outdir, filename)
    print(f"Saving model to: {outpath}")
    torch.save(model.state_dict(), outpath)
    return None
 
def load_model(model_path: str | None = None) -> SiameseNetwork:
    model = SiameseNetwork()

    if model_path is not None:
        print(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict) 
        return model
    
    latest_model_datetime = None
    latest_model_path = None
    for model_filename in os.listdir(models_dir):
        model_timestamp = float(model_filename.replace('.params', '').replace('siamese_', ''))
        model_datetime = datetime.fromtimestamp(model_timestamp)
        if latest_model_path is None or model_datetime > latest_model_datetime:
            latest_model_datetime = model_datetime
            latest_model_path = os.path.join(models_dir, model_filename)

    print(f"Loading model from: {latest_model_path}") 
    state_dict = torch.load(latest_model_path)
    model.load_state_dict(state_dict)

    return model

def train_model(model: SiameseNetwork, train_loader: DataLoader) -> None:

    train_loader.dataset.set_iter_pairwise(True)

    # Decalre Loss Function
    criterion = ContrastiveLoss()
    # Declare Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()

    print("#### STARTING TRANING #############################################################")    
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS+1):
        epoch_loss = 0
        for i, (img0, img1, label) in enumerate(train_loader):
            img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img0, img1)
            loss= criterion(output1, output2, label)
            loss.backward()
            optimizer.step()    

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}], Loss: {avg_loss:.5f}") 

    print("#### FINISHED TRANING #############################################################")    
    elapsed_time = time.time() - start_time
    print(f"Traning Took: {elapsed_time:3f}s or {(elapsed_time/60):.3f}mins")

    return None
