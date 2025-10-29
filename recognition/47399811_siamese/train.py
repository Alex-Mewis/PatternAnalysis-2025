"""
contains code for traning, validating, testing and saving the model.
"""
import os, time
import numpy as np

import torch
from torch import nn
from torch.nn import TripletMarginLoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score 

from modules import SiameseNetwork, Classifier, save_model
from configs import siamese_config, classifier_config
from plotting import plot_loss, plot_tsne, plot_classifer_traning_metrics

#### PERAMBLE #####################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

 
#### MAIN FUNCTIONS ###############################################################
def train_model(siamese: SiameseNetwork, train_loader: DataLoader, validation_loader: DataLoader) -> None:

    train_loader.dataset.set_triple_iter(True)
    validation_loader.dataset.set_triple_iter(True)

    triplet_loss = TripletMarginLoss().to(device)
    cross_entropy_loss = CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(siamese.parameters(), lr=siamese_config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=siamese_config.epochs, eta_min=1e-7) 

    training_metrics = {'loss': list(), 'acc': list(), 'auc-roc': list()}  
    validation_metrics = {'loss': list(), 'acc': list(), 'auc-roc': list()}  

    print("#### STARTING TRANING SIAMESE NETWORK #############################################")
    start_time = time.time()
    for epoch in range(1, siamese_config.epochs+1):

        train_features = torch.empty((0, 512)).to(device)
        validation_features = torch.empty((0, 512)).to(device)
        train_labels = torch.empty(0).to(device)
        validation_labels = torch.empty(0).to(device)
        
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

            # if epoch == siamese_config.epochs:
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
                
                # if epoch == siamese_config.epochs:
                validation_features = torch.cat([validation_features, anchor_features])
                validation_labels = torch.cat([validation_labels, label])

        validation_avg_loss = validation_epoch_loss / len(validation_loader)
        validation_metrics['loss'].append(validation_avg_loss)
        validation_metrics['acc'].append(accuracy_score(epoch_labels, epoch_preds))
        validation_metrics['auc-roc'].append(roc_auc_score(epoch_labels, epoch_probs))

        scheduler.step()
        
        plot_classifer_traning_metrics(training_metrics, validation_metrics)

        plot_tsne(train_features, train_labels, "Train")
        plot_tsne(validation_features, validation_labels, "Validation")

        save_model(siamese)

        print(f"Epoch [{epoch}/{siamese_config.epochs}], Training Loss: {training_avg_loss:.5f}, Validation Loss: {validation_avg_loss:.5f}") 

    print("#### FINISHED TRANING SIAMESE NETWORK #############################################")    
    elapsed_time = time.time() - start_time
    print(f"Traning Took: {elapsed_time:3f}s or {(elapsed_time/60):.3f}mins")


    return None

def train_classifer(classifier: Classifier, siamese: SiameseNetwork, train_loader: DataLoader, validation_loader: DataLoader) -> None:

    train_loader.dataset.set_triple_iter(False)
    validation_loader.dataset.set_triple_iter(False)

    cross_entropy_loss = CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(list(siamese.parameters()) + list(classifier.parameters()), lr=classifier_config.learning_rate)
    # scheduler = CosineAnnealingLR(optimizer, T_max=classifier_config.epochs, eta_min=1e-5) 
    
    # siamese.eval()

     
    training_metrics = {'loss': list(), 'acc': list(), 'auc-roc': list()}  
    validation_metrics = {'loss': list(), 'acc': list(), 'auc-roc': list()}  

    print("#### STARTED TRANING CLASSIFIER ###################################################")    
    start_time = time.time()
    for epoch in range(1, classifier_config.epochs+1):
        
        # train.
        classifier.train()
        siamese.train()
        training_epoch_loss = 0
        epoch_probs  = list()
        epoch_preds  = list()
        epoch_labels = list()
        for img, label in train_loader: 
            img, label = img.to(device), label.to(device).long()

            optimizer.zero_grad()
            
            #with torch.no_grad():
            latent_vector = siamese.forward_once(img)
            
            out = classifier(latent_vector)
            
            classifier_loss = cross_entropy_loss(out, label)


            loss.backward()
            optimizer.step()
            
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)

            training_epoch_loss += loss.item()
            epoch_probs.extend(probs.detach().cpu().numpy())
            epoch_preds.extend(preds.detach().cpu().numpy())
            epoch_labels.extend(label.cpu().numpy())

        training_avg_loss = training_epoch_loss / len(train_loader)
        training_metrics['loss'].append(training_avg_loss)
        training_metrics['acc'].append(accuracy_score(epoch_labels, epoch_preds))
        training_metrics['auc-roc'].append(roc_auc_score(epoch_labels, epoch_probs))

        # validate.
        siamese.eval()
        classifier.eval()
        validation_epoch_loss = 0
        epoch_probs  = list()
        epoch_preds  = list()
        epoch_labels = list()
        with torch.no_grad():
            for img, label in validation_loader:
                img, label = img.to(device), label.to(device).long()

                latent_vector = siamese.forward_once(img)
                out = classifier(latent_vector)
                loss = cross_entropy_loss(out, label)
            
                probs = torch.softmax(out, dim=1)[:, 1]
                preds = torch.argmax(out, dim=1)

                validation_epoch_loss += loss.item()
                epoch_probs.extend(probs.detach().cpu().numpy())
                epoch_preds.extend(preds.detach().cpu().numpy())
                epoch_labels.extend(label.cpu().numpy())

        validation_avg_loss = validation_epoch_loss / len(validation_loader)
        validation_metrics['loss'].append(validation_avg_loss)
        validation_metrics['acc'].append(accuracy_score(epoch_labels, epoch_preds))
        validation_metrics['auc-roc'].append(roc_auc_score(epoch_labels, epoch_probs))
        
        # scheduler.step()
        
        plot_classifer_traning_metrics(training_metrics, validation_metrics)
        save_model(siamese)
        save_model(classifier)

        # for name, param in classifier.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.abs().mean().item())

        print(f"Epoch [{epoch}/{classifier_config.epochs}], Training Loss: {training_avg_loss:.5f}, Validation Loss: {validation_avg_loss:.5f}")


    print("#### FINISHED TRANING CLASSIFIER ##################################################")   
    elapsed_time = time.time() - start_time
    print(f"Traning Took: {elapsed_time:.3f}s or {(elapsed_time/60):.3f}mins")


    return None

def test_accuracy(siamese: SiameseNetwork, classifier: Classifier, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:

    test_loader.dataset.set_triple_iter(False)

    siamese.eval()
    classifier.eval()

    all_labels = np.zeros(len(test_loader.dataset)) 
    all_probs = np.zeros(len(test_loader.dataset))
    all_preds = np.zeros(len(test_loader.dataset))
    
    n = 0

    print("#### STARTED TESTING ACCURACY #####################################################")  
    with torch.no_grad():
        num_correct = 0
        total = 0

        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = len(labels)

            latent_vector = siamese.forward_once(imgs)
            probs = classifier(latent_vector)
            preds = torch.round(probs)

            all_labels[n:n+batch_size] = labels.cpu().numpy()
            all_probs[n:n+batch_size] = probs.cpu().numpy()
            all_preds[n:n+batch_size] = preds.cpu().numpy()
            
            n += batch_size 

            total += len(probs)
            num_correct += (preds == labels).sum().item()
        
    print(f"Testing Accuracy: {(100*num_correct/total):.2f}%")
    print(f"AUC-ROC score   : {roc_auc_score(all_labels, all_probs):.2f}")

    print("#### FINISHED TESTING ACCURACY ####################################################")  

    return all_labels, all_probs, all_preds 
