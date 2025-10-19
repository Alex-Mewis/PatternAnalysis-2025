"""
Contains all of the relavent functions for making plots.
"""
import os
from datetime import datetime
import numpy as np
import torch
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import seaborn as sns

#### PERAMBLE #####################################################################
this_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(this_dir, "plots")
train_plots_dir = os.path.join(plots_dir, "train")

if not os.path.exists(plots_dir): os.mkdir(plots_dir)
if not os.path.exists(train_plots_dir): os.mkdir(train_plots_dir)

#### FUNCTIONS ####################################################################
def plot_loss(training_loss: list[float], validation_loss: list[float], model_title: str) -> None:
    """
    """
    model_name = model_title.lower().replace(' ', '_')
    outpath = os.path.join(train_plots_dir, f"{model_name}_{datetime.now().timestamp()}.png")

    epochs = list(range(1, len(training_loss)+1))   
    plt.plot(epochs, training_loss, label="Traning")
    plt.plot(epochs, validation_loss, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss") 
   
    plt.savefig(outpath)
    print(f"Saved: {outpath}") 
    
    plt.close()
    return None


def plot_tsne(features: torch.Tensor, labels: torch.Tensor) -> None:

    features, labels = features.cpu().numpy(), labels.cpu().numpy()
    tsne = TSNE()
    tsne_output = tsne.fit_transform(features)

    benign_tsne = tsne_output[labels == 0, :]
    malignant_tsne = tsne_output[labels == 1, :]

    plt.scatter(
        benign_tsne[:,0],
        benign_tsne[:,1],
        c='blue',
        label="Benign",
    )

    plt.scatter(
        x=malignant_tsne[:,0],
        y=malignant_tsne[:,1],
        c='red',
        label="Malignant",
    )

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title("TSNE Scatter")
    
    outpath = os.path.join(plots_dir, "TSNE Scatter")
    plt.savefig(outpath)
    print(f"Saved: {outpath}")

    plt.close()
    return None

def plot_confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> None:
    """
    """
    get_num_correct   = lambda label : np.sum((predictions == labels) & (labels == label))
    get_num_incorrect = lambda label : np.sum((predictions != labels) & (labels == label)) 

    confusion_matrix = [[get_num_correct(1)  , get_num_incorrect(0)], 
                        [get_num_incorrect(1), get_num_correct(0)  ]]

    sns.heatmap(confusion_matrix, xticklabels=['Actually Malignant', 'Actually Benign'],
                yticklabels=['Preidcted Malignant', 'Predicted Bengin'], annot=True)
    
    plt.title(f"Confusion Matrix")

    outpath = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(outpath)

    print(f"Saved: {outpath}")
    plt.close()
    return None