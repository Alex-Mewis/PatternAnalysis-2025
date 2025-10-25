"""
contains code for traning, validating, testing and saving the model.
"""
import os, time
from datetime import datetime
import numpy as np

import torch
from torch import nn
from torch.nn import TripletMarginLoss, CrossEntropyLoss
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score 

from modules import SiameseNetwork, Classifier
from configs import siamese_config, classifier_config
from plotting import plot_loss, plot_tsne, plot_classifer_traning_metrics

#### PERAMBLE #####################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
this_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(this_dir, "models")
siamese_models_dir = os.path.join(models_dir, 'siamese')
classifier_models_dir = os.path.join(models_dir, 'classifier')

if not os.path.exists(models_dir): os.mkdir(models_dir)
if not os.path.exists(siamese_models_dir): os.mkdir(siamese_models_dir)
if not os.path.exists(classifier_models_dir): os.mkdir(classifier_models_dir)


#### MODEL FUNCTIONS #############################################################
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
    
    if model_name.lower() == 'siamese':
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

def train_model(siamese: SiameseNetwork, train_loader: DataLoader, validation_loader: DataLoader) -> None:

    train_loader.dataset.set_triple_iter(True)
    validation_loader.dataset.set_triple_iter(True)

    triplet_loss = TripletMarginLoss() 
    optimizer = torch.optim.Adam(siamese.parameters(), lr=siamese_config.learning_rate)
    
    training_epoch_losses = list()
    validation_epoch_losses = list()

    train_features = torch.empty((0, 512)).to(device)
    validation_features = torch.empty((0, 512)).to(device)
    train_labels = torch.empty(0).to(device)
    validation_labels = torch.empty(0).to(device)

    print("#### STARTING TRANING SIAMESE NETWORK #############################################")
    start_time = time.time()
    for epoch in range(1, siamese_config.epochs+1):

        # train. 
        siamese.train()
        training_epoch_loss = 0
        for anchor_img, positive_img, negative_img, label in train_loader:
            anchor_img, positive_img = anchor_img.to(device), positive_img.to(device)
            negative_img, label      = negative_img.to(device), label.to(device) 

            optimizer.zero_grad()
            
            anchor_features, positive_features, negative_features = siamese([anchor_img, positive_img, negative_img])

            loss = triplet_loss(anchor_features, positive_features, negative_features)
            
            loss.backward()
            optimizer.step()    

            training_epoch_loss += loss.item()

            if epoch == siamese_config.epochs:
                train_features = torch.cat([train_features, anchor_features])
                train_labels = torch.cat([train_labels, label])

        training_avg_loss = training_epoch_loss / len(train_loader)
        training_epoch_losses.append(training_avg_loss)

        # validate. 
        siamese.eval()
        validation_epoch_loss = 0
        with torch.no_grad():
            for anchor_img, positive_img, negative_img, label in validation_loader:
                anchor_img, positive_img = anchor_img.to(device), positive_img.to(device)
                negative_img, label      = negative_img.to(device), label.to(device) 
            
                anchor_features, positive_features, negative_features = siamese([anchor_img, positive_img, negative_img])
                loss = triplet_loss(anchor_features, positive_features, negative_features)
               
                validation_epoch_loss += loss.item() 

                if epoch == siamese_config.epochs:
                    validation_features = torch.cat([validation_features, anchor_features])
                    validation_labels = torch.cat([validation_labels, label])

        validation_avg_loss = validation_epoch_loss / len(validation_loader)
        validation_epoch_losses.append(validation_avg_loss)


        print(f"Epoch [{epoch}/{siamese_config.epochs}], Training Loss: {training_avg_loss:.5f}, Validation Loss: {validation_avg_loss:.5f}") 

    print("#### FINISHED TRANING SIAMESE NETWORK #############################################")    
    elapsed_time = time.time() - start_time
    print(f"Traning Took: {elapsed_time:3f}s or {(elapsed_time/60):.3f}mins")

    plot_loss(training_epoch_losses, validation_epoch_losses, "Siamese Network")

    plot_tsne(train_features, train_labels, "Train")
    plot_tsne(validation_features, validation_labels, "Validation")

    return None

def train_classifer(classifier: Classifier, siamese: SiameseNetwork, train_loader: DataLoader, validation_loader: DataLoader) -> None:

    train_loader.dataset.set_triple_iter(False)
    validation_loader.dataset.set_triple_iter(False)

    cross_entropy_loss = CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_config.learning_rate)

    siamese.eval()

     
    training_metrics = {'loss': list(), 'acc': list(), 'auc-roc': list()}  
    validation_metrics = {'loss': list(), 'acc': list(), 'auc-roc': list()}  

    print("#### STARTED TRANING CLASSIFIER ###################################################")    
    start_time = time.time()
    for epoch in range(1, classifier_config.epochs+1):
        
        # train.
        classifier.train()
        training_epoch_loss = 0
        epoch_probs  = list()
        epoch_preds  = list()
        epoch_labels = list()
        for img, label in train_loader: 
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()

            latent_vector = siamese.forward_once(img)
            probs = classifier(latent_vector)
            preds = torch.round(probs) 

            loss = cross_entropy_loss(probs, label)

            loss.backward()
            optimizer.step()

            training_epoch_loss += loss.item()
            epoch_probs.extend(probs.detach().cpu().numpy())
            epoch_preds.extend(preds.detach().cpu().numpy())
            epoch_labels.extend(label.cpu().numpy())

        training_avg_loss = training_epoch_loss / len(train_loader)
        training_metrics['loss'].append(training_avg_loss)
        training_metrics['acc'].append(accuracy_score(epoch_labels, epoch_preds))
        training_metrics['auc-roc'].append(roc_auc_score(epoch_labels, epoch_probs))

        # validate.
        classifier.eval()
        validation_epoch_loss = 0
        epoch_probs  = list()
        epoch_preds  = list()
        epoch_labels = list()
        with torch.no_grad():
            for img, label in validation_loader:
                img, label = img.to(device), label.to(device)

                latent_vector = siamese.forward_once(img)
                probs = classifier(latent_vector)
                preds = torch.round(probs) 


                loss = cross_entropy_loss(probs, label)

                validation_epoch_loss += loss.item()
                epoch_probs.extend(probs.detach().cpu().numpy())
                epoch_preds.extend(preds.detach().cpu().numpy())
                epoch_labels.extend(label.cpu().numpy())
        
        validation_avg_loss = validation_epoch_loss / len(validation_loader)
        validation_metrics['loss'].append(validation_avg_loss)
        validation_metrics['acc'].append(accuracy_score(epoch_labels, epoch_preds))
        validation_metrics['auc-roc'].append(roc_auc_score(epoch_labels, epoch_probs))

        print(f"Epoch [{epoch}/{classifier_config.epochs}], Training Loss: {training_avg_loss:.5f}, Validation Loss: {validation_avg_loss:.5f}")


    print("#### FINISHED TRANING CLASSIFIER ##################################################")   
    elapsed_time = time.time() - start_time
    print(f"Traning Took: {elapsed_time:.3f}s or {(elapsed_time/60):.3f}mins")

    plot_classifer_traning_metrics(training_metrics, validation_metrics)

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
