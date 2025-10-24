"""
Contains all of the relavent functions for making plots.
"""
import os
from datetime import datetime
import numpy as np
import torch
from sklearn.manifold import TSNE

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
import seaborn as sns
import catppuccin

#### PERAMBLE #####################################################################
matplotlib.style.use("mocha")
mocha_colours = catppuccin.PALETTE.mocha.colors

catpuccin_cmap = LinearSegmentedColormap.from_list("catpuccin_cmap",
                [mocha_colours.mauve.hex, mocha_colours.red.hex, mocha_colours.maroon.hex, mocha_colours.peach.hex, 
                 mocha_colours.yellow.hex, mocha_colours.rosewater.hex],
                N=256)

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
    plt.plot(epochs, training_loss, label="Training")
    plt.plot(epochs, validation_loss, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss") 
    plt.title(f"Loss of {model_title}")
    plt.grid()
    plt.gca().set_facecolor(mocha_colours.mantle.hex)
   
    plt.savefig(outpath)
    print(f"Saved: {outpath}") 
    
    plt.close()
    return None


def plot_tsne(features: torch.Tensor, labels: torch.Tensor, dataset: str) -> None:

    features, labels = features.cpu().detach().numpy(), labels.cpu().detach().numpy()
    tsne = TSNE()
    tsne_output = tsne.fit_transform(features)

    benign_tsne = tsne_output[labels == 0, :]
    malignant_tsne = tsne_output[labels == 1, :]

    plt.scatter(
        benign_tsne[:,0],
        benign_tsne[:,1],
        label="Benign",
        edgecolor=mocha_colours.text.hex,
        linewidths=0.7,
        alpha=0.5,
    )

    plt.scatter(
        x=malignant_tsne[:,0],
        y=malignant_tsne[:,1],
        label="Malignant",
        edgecolor=mocha_colours.text.hex,
        linewidths=0.3,
        alpha=0.5,
    )

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title(f"TSNE Scatter on {dataset} dataset")
    plt.gca().set_facecolor(mocha_colours.mantle.hex)
    
    outpath = os.path.join(plots_dir, f"tsne_scatter_{dataset.lower()}.png")
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
                yticklabels=['Preidcted Malignant', 'Predicted Bengin'], annot=True, cmap=catpuccin_cmap, fmt='d')
    
    plt.title(f"Confusion Matrix")

    outpath = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(outpath)

    print(f"Saved: {outpath}")
    plt.close()
    return None