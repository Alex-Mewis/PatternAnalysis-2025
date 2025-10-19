"""
contains code for traning, validating, testing and saving the model.
"""
import os, time
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from modules import SiameseNetwork, Classifier
from configs import siamese_config, classifier_config
from plotting import plot_loss, plot_tsne 

#### PERAMBLE #####################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
this_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(this_dir, "models")
siamese_models_dir = os.path.join(models_dir, 'siamese')
classifier_models_dir = os.path.join(models_dir, 'classifier')

if not os.path.exists(models_dir): os.mkdir(models_dir)
if not os.path.exists(siamese_models_dir): os.mkdir(siamese_models_dir)
if not os.path.exists(classifier_models_dir): os.mkdir(classifier_models_dir)

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
def save_model(model: nn.Module, outdir: str | None = None) -> None:
    if outdir is None: outdir = os.path.join(models_dir, model.name)
    filename = f"{model.name}_{datetime.now().timestamp()}.params"
    outpath = os.path.join(outdir, filename)
    print(f"Saving model to: {outpath}")
    torch.save(model.state_dict(), outpath)
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

    train_loader.dataset.set_iter_pairwise(True)
    validation_loader.dataset.set_iter_pairwise(True)

    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(siamese.parameters(), lr=siamese_config.learning_rate)
    
    training_epoch_losses = list()
    validation_epoch_losses = list()
    
    features = torch.empty((0, 1000)).to(device)
    labels = torch.empty(0).to(device)

    print("#### STARTING TRANING SIAMESE NETWORK #############################################")
    start_time = time.time()
    for epoch in range(1, siamese_config.epochs+1):

        # train. 
        siamese.train()
        training_epoch_loss = 0
        for img0, img1, label in train_loader:
            img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)
            
            optimizer.zero_grad()
            
            output1, output2 = siamese(img0, img1)
            loss = criterion(output1, output2, label)
            
            loss.backward()
            optimizer.step()    

            training_epoch_loss += loss.item()

        training_avg_loss = training_epoch_loss / len(train_loader)
        training_epoch_losses.append(training_avg_loss)

        # validate. 
        siamese.eval()
        validation_epoch_loss = 0
        with torch.no_grad():
            for img0, img1, label in validation_loader:
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                out0, out1 = siamese(img0, img1) 
                loss = criterion(out0, out1, label)

                validation_epoch_loss += loss.item() 

                if epoch == siamese_config.epochs:
                    features = torch.cat([features, out0])
                    labels = torch.cat([labels, label])

        validation_avg_loss = validation_epoch_loss / len(validation_loader)
        validation_epoch_losses.append(validation_avg_loss)


        print(f"Epoch [{epoch}/{siamese_config.epochs}], Traning Loss: {training_avg_loss:.6g}, Validation Loss: {validation_avg_loss:.6g}") 

    print("#### FINISHED TRANING SIAMESE NETWORK #############################################")    
    elapsed_time = time.time() - start_time
    print(f"Traning Took: {elapsed_time:3f}s or {(elapsed_time/60):.3f}mins")

    plot_loss(training_epoch_losses, validation_epoch_losses, "Siamese Network")

    plot_tsne(features, labels) 

    return None

def train_classifer(classifier: Classifier, siamese: SiameseNetwork, train_loader: DataLoader, validation_loader: DataLoader) -> None:

    train_loader.dataset.set_iter_pairwise(False)
    validation_loader.dataset.set_iter_pairwise(False)

    cross_entropy_loss = CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_config.learning_rate)

    siamese.eval()
    
    training_epoch_losses = list()
    validation_epoch_losses = list()

    print("#### STARTED TRANING CLASSIFIER ###################################################")    
    start_time = time.time()
    for epoch in range(1, classifier_config.epochs+1):
        
        # train.
        classifier.train()
        training_epoch_loss = 0
        for img, label in train_loader: 
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()

            latent_vector = siamese.forward_once(img)
            out = classifier(latent_vector)

            loss = cross_entropy_loss(out, label)
            loss.backward()
            optimizer.step()

            training_epoch_loss += loss.item()

        training_avg_loss = training_epoch_loss / len(train_loader)
        training_epoch_losses.append(training_avg_loss)

        # validate.
        classifier.eval()
        validation_epoch_loss = 0
        with torch.no_grad():
            for img, label in validation_loader:
                img, label = img.to(device), label.to(device)

                latent_vector = siamese.forward_once(img)
                out = classifier(latent_vector)

                loss = cross_entropy_loss(out, label)

                validation_epoch_loss += loss.item()

        validation_avg_loss = validation_epoch_loss / len(validation_loader)
        validation_epoch_losses.append(validation_avg_loss)        

        print(f"Epoch [{epoch}/{classifier_config.epochs}], Training Loss: {training_avg_loss:.6g}, Validation Loss: {validation_avg_loss:.6g}")


    print("#### FINISHED TRANING CLASSIFIER ##################################################")   
    elapsed_time = time.time() - start_time
    print(f"Traning Took: {elapsed_time:.3f}s or {(elapsed_time/60):.3f}mins")

    plot_loss(training_epoch_losses, validation_epoch_losses, "Binary Classifier")

    return None

def test_accuracy(siamese: SiameseNetwork, classifier: Classifier, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:

    test_loader.dataset.set_iter_pairwise(False)

    siamese.eval()
    classifier.eval()

    total_predictions = np.zeros(len(test_loader.dataset))
    total_labels = np.zeros(len(test_loader.dataset)) 
    n = 0

    print("#### STARTED TESTING ACCURACY #####################################################")  
    with torch.no_grad():
        num_correct = 0
        total = 0

        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = len(labels)

            latent_vector = siamese.forward_once(imgs)
            out = classifier(latent_vector)
            pred = torch.round(out)

            total_predictions[n:n+batch_size] = pred.cpu().numpy() 
            total_labels[n:n+batch_size] = labels.cpu().numpy()
            n += batch_size 

            total += len(out)
            num_correct += (pred == labels).sum().item()
        
        print(f"Testing Accuracy: {(100*num_correct/total):.2f}%")

    print("#### FINISHED TESTING ACCURACY ####################################################")  

    return total_predictions, total_labels 
