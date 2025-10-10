"""
The script which is run.
"""
import os

from dataset import ISICImageDataset

this_dir = os.path.dirname(os.path.abspath(__file__))

#### INPUTS ############################################################################
## DATASETS ################################################
IMAGE_DIR = os.path.join(this_dir, 'data', 'chats_cats')
LABELS_PATH = os.path.join(this_dir, 'data', 'ISIC_2020_Training_GroundTruth.csv')

## HYPERPARAMETERS #########################################
BATCH_SIZE = 64


def main() -> None:
    dataset = ISICImageDataset(IMAGE_DIR, LABELS_PATH) 
    train_dataset, test_dataset = dataset.split()   
    train_dataloader = train_dataset.to_DataLoader(batch_size=BATCH_SIZE)
    test_dataloader = test_dataset.to_DataLoader(batch_size=BATCH_SIZE)


if __name__ == "__main__":
    main()