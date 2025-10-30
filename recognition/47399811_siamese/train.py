"""
contains code for traning, validating, testing and saving the model.
"""
import time
import numpy as np

import torch
from torch.nn import TripletMarginLoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score 

from modules import SiameseNetwork, save_model 
from plotting import plot_loss, plot_tsne 
from dataset import get_train_validation_test_dataloaders

#### PARAMETERS ###################################################################
EPOCHS = 16
LEARNING_RATE = 0.01

#### PERAMBLE #####################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

 
#### MAIN FUNCTIONS ###############################################################
def train_model(siamese: SiameseNetwork, train_loader: DataLoader, validation_loader: DataLoader) -> None:

    train_loader.dataset.set_triple_iter(True)
    validation_loader.dataset.set_triple_iter(True)

    triplet_loss = TripletMarginLoss().to(device)
    cross_entropy_loss = CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(siamese.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-7) 

    training_metrics = {'loss': list(), 'acc': list(), 'auc-roc': list()}  
    validation_metrics = {'loss': list(), 'acc': list(), 'auc-roc': list()}  
    train_labels = torch.empty(0).to(device)
    validation_labels = torch.empty(0).to(device)

    print("#### STARTING TRANING SIAMESE NETWORK #############################################")
    start_time = time.time()
    for epoch in range(1, EPOCHS+1):

        train_features = torch.empty((0, 512)).to(device)
        validation_features = torch.empty((0, 512)).to(device)
        
        # train. 
        siamese.train()
        training_epoch_loss = 0
        epoch_probs  = list()
        epoch_preds  = list()
        epoch_labels = list()
        for anchor_img, positive_img, negative_img, label in train_loader:
            anchor_img, positive_img = anchor_img.to(device), positive_img.to(device)
            negative_img, label      = negative_img.to(device), label.to(device).long()

            optimizer.zero_grad()
            
            anchor_features, positive_features, negative_features = siamese([anchor_img, positive_img, negative_img])
            siamese_loss = triplet_loss(anchor_features, positive_features, negative_features)
            
            classifier_out = siamese.classify(anchor_img)
            classifier_loss = cross_entropy_loss(classifier_out, label)
            probs = torch.softmax(classifier_out, dim=1)[:, 1]
            preds = torch.argmax(classifier_out, dim=1)

            loss = siamese_loss + classifier_loss

            loss.backward()
            optimizer.step()    

            training_epoch_loss += loss.item()
            epoch_probs.extend(probs.detach().cpu().numpy())
            epoch_preds.extend(preds.detach().cpu().numpy())
            epoch_labels.extend(label.cpu().numpy())

            if epoch == EPOCHS:
                train_features = torch.cat([train_features, anchor_features])
                train_labels = torch.cat([train_labels, label])

        training_avg_loss = training_epoch_loss / len(train_loader)
        training_metrics['loss'].append(training_avg_loss)
        training_metrics['acc'].append(accuracy_score(epoch_labels, epoch_preds))
        training_metrics['auc-roc'].append(roc_auc_score(epoch_labels, epoch_probs))

        # validate. 
        siamese.eval()
        validation_epoch_loss = 0
        epoch_probs  = list()
        epoch_preds  = list()
        epoch_labels = list()
        with torch.no_grad():
            for anchor_img, positive_img, negative_img, label in validation_loader:
                anchor_img, positive_img = anchor_img.to(device), positive_img.to(device)
                negative_img, label      = negative_img.to(device), label.to(device).long()
            
                anchor_features, positive_features, negative_features = siamese([anchor_img, positive_img, negative_img])
                siamese_loss = triplet_loss(anchor_features, positive_features, negative_features)
            
                classifier_out = siamese.classify(anchor_img)
                classifier_loss = cross_entropy_loss(classifier_out, label)
                probs = torch.softmax(classifier_out, dim=1)[:, 1]
                preds = torch.argmax(classifier_out, dim=1)

                loss = siamese_loss + classifier_loss
                validation_epoch_loss += loss.item() 

                epoch_probs.extend(probs.detach().cpu().numpy())
                epoch_preds.extend(preds.detach().cpu().numpy())
                epoch_labels.extend(label.cpu().numpy())
                
                if epoch == EPOCHS:
                    validation_features = torch.cat([validation_features, anchor_features])
                    validation_labels = torch.cat([validation_labels, label])

        validation_avg_loss = validation_epoch_loss / len(validation_loader)
        validation_metrics['loss'].append(validation_avg_loss)
        validation_metrics['acc'].append(accuracy_score(epoch_labels, epoch_preds))
        validation_metrics['auc-roc'].append(roc_auc_score(epoch_labels, epoch_probs))

        scheduler.step()
        
        print(f"Epoch [{epoch}/{EPOCHS.epochs}], Training Loss: {training_avg_loss:.5f}, Validation Loss: {validation_avg_loss:.5f}")
        print(f"             , Training Accuracy: {training_metrics['acc'][-1]}, Validation Accuracy: {validation_metrics['acc'][-1]}") 
        print(f"             , Training AUC-ROC: {training_metrics['auc-roc'][-1]}, Validation AUC-ROC: {validation_metrics['AUC-ROC'][-1]}") 

    # make plots   
    plot_loss(training_metrics, validation_metrics)
    plot_tsne(train_features, train_labels, "Train")
    plot_tsne(validation_features, validation_labels, "Validation")

    print("#### FINISHED TRANING SIAMESE NETWORK #############################################")    
    elapsed_time = time.time() - start_time
    print(f"Traning Took: {elapsed_time:3f}s or {(elapsed_time/60):.3f}mins")


    return None


if __name__ == "__main__":
    
    train_dataloader, validation_dataloader, _ = get_train_validation_test_dataloaders()
    siamese = SiameseNetwork().to(device)
    train_model(siamese, train_dataloader, validation_dataloader)
    save_model(siamese)
     