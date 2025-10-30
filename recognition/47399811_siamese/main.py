"""
The script which is run.
"""
import os
import torch

from dataset import get_train_validation_test_dataloaders 
from modules import SiameseNetwork, load_model, save_model
from train import train_model
from plotting import plot_confusion_matrix, plot_roc_curve
from predict import test_accuracy

this_dir = os.path.dirname(os.path.abspath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu': print("Warning using CPU!")

#### INPUTS ############################################################################
## CONTROL FLOW ############################################
LOAD_MOST_RECENT_MODEL = True 

TRAIN_SIAMESE = False 
TEST_ACCURACY = True 

MAKE_PREDICTION_PLOTS = True 

## DATASETS ################################################D
IMAGE_DIR = os.path.join(this_dir, 'data', 'images')
LABELS_PATH = os.path.join(this_dir, 'data', 'ISIC_2020_Training_GroundTruth.csv')

## HYPERPARAMETERS #########################################D
BATCH_SIZE = 32 


def main() -> None:

    dataloaders = get_train_validation_test_dataloaders(IMAGE_DIR, LABELS_PATH)
    train_dataloader, validation_dataloader, test_dataloader = dataloaders 
    
    siamese = SiameseNetwork() if not LOAD_MOST_RECENT_MODEL else load_model()
    
    siamese = siamese.to(device)

    if TRAIN_SIAMESE: 
        train_model(siamese, train_dataloader, validation_dataloader) 
        save_model(siamese)


    if TEST_ACCURACY:
       labels, probabilities, predictions = test_accuracy(siamese, test_dataloader)

    if MAKE_PREDICTION_PLOTS:
        plot_confusion_matrix(predictions, labels)
        plot_roc_curve(probabilities, labels) 


if __name__ == "__main__":
    main()