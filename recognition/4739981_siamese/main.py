"""
The script which is run.
"""
import os
import torch

from dataset import ISICImageDataset
from modules import SiameseNetwork
from train import train_model

this_dir = os.path.dirname(os.path.abspath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu': print("Warning using CPU!")

#### INPUTS ############################################################################
## DATASETS ################################################
IMAGE_DIR = os.path.join(this_dir, 'data', 'images')
LABELS_PATH = os.path.join(this_dir, 'data', 'ISIC_2020_Training_GroundTruth.csv')

## HYPERPARAMETERS #########################################
BATCH_SIZE = 2 


def main() -> None:
    dataset = ISICImageDataset(IMAGE_DIR, LABELS_PATH) 
    train_dataset, test_dataset = dataset.split()   
    train_dataloader = train_dataset.to_DataLoader(batch_size=BATCH_SIZE)
    test_dataloader = test_dataset.to_DataLoader(batch_size=BATCH_SIZE)

    model = SiameseNetwork()
    model.to(device)
    train_model(model, train_dataloader) 


if __name__ == "__main__":
    main()