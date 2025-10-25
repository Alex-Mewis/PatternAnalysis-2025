"""
The script which is run.
"""
import os
import torch

from dataset import ISICImageDataset
from modules import SiameseNetwork, Classifier
from train import train_model, train_classifer, save_model, load_model, test_accuracy
from plotting import plot_confusion_matrix, plot_roc_curve

this_dir = os.path.dirname(os.path.abspath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu': print("Warning using CPU!")

#### INPUTS ############################################################################
## CONTROL FLOW ############################################
LOAD_MOST_RECENT_SIAMESE = True 
LOAD_MOST_RECENT_CLASSIFIER = False 

TRAIN_SIAMESE = False 
TRAIN_CLASSIFIER = True  
TEST_ACCURACY = True 

MAKE_PREDICTION_PLOTS = True 

## DATASETS ################################################
IMAGE_DIR = os.path.join(this_dir, 'data', 'images')
LABELS_PATH = os.path.join(this_dir, 'data', 'ISIC_2020_Training_GroundTruth.csv')

## HYPERPARAMETERS #########################################
BATCH_SIZE = 32 


def main() -> None:
    dataset = ISICImageDataset(IMAGE_DIR, LABELS_PATH) 

    # split the data set into 70% train, 15% validation and 15% test.
    train_dataset, hidden_dataset = dataset.split(p=0.7)   
    validation_dataset, test_dataset = hidden_dataset.split(p=0.5)
    
    train_dataset.force_even_data()
    train_dataset.set_training_data_transforms()
    # validation_dataset.force_even_data()
    
    train_dataloader = train_dataset.to_DataLoader(batch_size=BATCH_SIZE)
    validation_dataloader = validation_dataset.to_DataLoader(batch_size=BATCH_SIZE)
    test_dataloader = test_dataset.to_DataLoader(batch_size=BATCH_SIZE)

    
    siamese = SiameseNetwork() if not LOAD_MOST_RECENT_SIAMESE else load_model('siamese')
    classifier = Classifier() if not LOAD_MOST_RECENT_CLASSIFIER else load_model('classifier')
    
    siamese = siamese.to(device)
    classifier = classifier.to(device)

    if TRAIN_SIAMESE: 
        train_model(siamese, train_dataloader, validation_dataloader) 
        save_model(siamese)

    if TRAIN_CLASSIFIER: 
        train_classifer(classifier, siamese, train_dataloader, validation_dataloader)
        save_model(classifier)

    if TEST_ACCURACY:
        labels, probabilities, predictions = test_accuracy(siamese, classifier, test_dataloader)

    if MAKE_PREDICTION_PLOTS:
        plot_confusion_matrix(predictions, labels)
        plot_roc_curve(probabilities, labels) 


if __name__ == "__main__":
    main()