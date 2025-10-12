"""
The script which is run.
"""
import os
import torch

from dataset import ISICImageDataset
from modules import SiameseNetwork
from train import train_model, save_model, load_model

this_dir = os.path.dirname(os.path.abspath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu': print("Warning using CPU!")

#### INPUTS ############################################################################
## CONTROL FLOW ############################################
LOAD_MOST_RECENT_MODEL = True 

TRAIN = True
TEST = False

## DATASETS ################################################
IMAGE_DIR = os.path.join(this_dir, 'data', 'images')
LABELS_PATH = os.path.join(this_dir, 'data', 'ISIC_2020_Training_GroundTruth.csv')

## HYPERPARAMETERS #########################################
BATCH_SIZE = 32 


def main() -> None:
    dataset = ISICImageDataset(IMAGE_DIR, LABELS_PATH) 
    train_dataset, test_dataset = dataset.split()   
    train_dataloader = train_dataset.to_DataLoader(batch_size=BATCH_SIZE)
    test_dataloader = test_dataset.to_DataLoader(batch_size=BATCH_SIZE)
    
    model = SiameseNetwork() if not LOAD_MOST_RECENT_MODEL else load_model()
    model.to(device)

    if TRAIN: 
        train_model(model, train_dataloader) 
        save_model(model)

if __name__ == "__main__":
    main()