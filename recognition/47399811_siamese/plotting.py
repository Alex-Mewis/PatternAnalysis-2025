"""
Contains all of the relevant functions for making all plots.

Made by: Alexander Mewis
"""
import os
from datetime import datetime
import numpy as np

import torch
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
import seaborn as sns
import catppuccin

#### PERAMBLE #####################################################################
matplotlib.style.use("mocha")
mocha_colours = catppuccin.PALETTE.mocha.colors

# cmap used in the confusion matrix plot.
catpuccin_cmap = LinearSegmentedColormap.from_list("catpuccin_cmap",
                [mocha_colours.mauve.hex, mocha_colours.red.hex, mocha_colours.maroon.hex, mocha_colours.peach.hex, 
                 mocha_colours.yellow.hex, mocha_colours.rosewater.hex],
                N=256)

plots_dir = os.path.join(os.getcwd(), "plots")
train_plots_dir = os.path.join(plots_dir, "train")

if not os.path.exists(plots_dir): os.mkdir(plots_dir)
if not os.path.exists(train_plots_dir): os.mkdir(train_plots_dir)

#### DATA SHOWCASE PLOTS ###############################################################
def plot_image_showcase(image_matricies: list[np.ndarray], labels: list[int], filename: str,
                        title: str | None = None, cams: list[np.ndarray] | None = None,
                        captions: list[tuple[str, bool]] | None = None) -> None:
    """
    Plots a 3x3 grid layout of 9 images. Adding the approbate formatting and styling as specified.

    Parameters:
        image_matricies [list]: a list of length 9 which contains the images pixel values.
        labels [list]: the label for each image 0 => benign and 1 => malignant.
        filename [str]: the filename which the plot will be saved under.
        title [str]: if not None adds this as the super title of the figure.
        cams [list]: if not None adds this as a gray-scale overlay on top of each image.
        captions [list]: a list of titles for each axis and text will be green if the bool 
            if True otherwise red. 
    """
    
    assert len(image_matricies) == 9
    
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, img_matrix in enumerate(image_matricies):
        axes[i%3, i//3].imshow(img_matrix)
        if cams is not None: axes[i%3, i//3].imshow(cams[i], cmap='plasma', alpha=0.5)
        axes[i%3, i//3].set_xticks([])
        axes[i%3, i//3].set_yticks([])
        axes[i%3, i//3].set_title("Malignant" if labels[i] else "Benign")
        if captions is not None: axes[i%3, i//3].set_title(captions[i][0], fontsize=10,
          color=mocha_colours.green.hex if captions[i][1] else mocha_colours.red.hex)

    if title is not None: fig.suptitle(title, fontsize=20)
    plt.tight_layout()

    outpath = os.path.join(plots_dir, f"{filename}.png")
    plt.savefig(outpath)
    print(f"Saved: {outpath}") 

    plt.close()
    return None


#### TRAINING PLOTS ####################################################################
def plot_loss(training_metrics: dict, validation_metrics: dict) -> None:
    """
    Plots the loss, Accuracy and AUC-ROC of the model on both training and validation data
    across all epochs.

    Parameters:
        training_metics: contains the models loss, accuracy and auc-roc score for all
            epochs on the training data. 
        validation_metrics: contains the models loss, accuracy and auc-roc score for all
            epochs on the training data. 
    """ 
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))    

    epochs = range(1, 1+len(training_metrics['loss'])) 

    axes[0].set_title("Traning & Validation Loss")
    axes[0].plot(epochs, training_metrics['loss'], label='Training')
    axes[0].plot(epochs, validation_metrics['loss'], label='Validation')
    axes[0].set_ylabel("Loss")

    axes[1].set_title("Traning & Validation Accuracy")
    axes[1].plot(epochs, training_metrics['acc'], label='Training')
    axes[1].plot(epochs, validation_metrics['acc'], label='Validation')
    axes[1].set_ylabel("Accuracy")

    axes[2].set_title("Traning & Validation AUC-ROC")
    axes[2].plot(epochs, training_metrics['auc-roc'], label='Training')
    axes[2].plot(epochs, validation_metrics['auc-roc'], label='Validation')
    axes[2].set_ylabel("AUC-ROC")
    
    for i in range(3):
        axes[i].set_xlabel("Epoch")
        axes[i].set_facecolor(mocha_colours.mantle.hex)
        axes[i].legend()
        axes[i].grid()

    plt.tight_layout()

    outpath = os.path.join(train_plots_dir, f"model_{datetime.now().timestamp()}.png")
    plt.savefig(outpath)
    print(f"Saved: {outpath}")

    plt.close()
    return None


def plot_tsne(features: torch.Tensor, labels: torch.Tensor, dataset: str) -> None: 
    """
    Plots the t-SNE scatter.

    Parameters:
        features [torch.Tensor]: the latent vectors of the model.
        label [torch.Tensor]: the labels for the latent vectors of the model.
        dataset [str]: the name of the dataset the plot is being made for.
    """
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
    
    outpath = os.path.join(train_plots_dir, f"tsne_scatter_{dataset.lower()}_{datetime.now().timestamp()}.png")
    plt.savefig(outpath)
    print(f"Saved: {outpath}")

    plt.close()
    return None


#### EVALUATION PLOTS ##################################################################
def plot_confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> None:
    """
    Plots the confusion matrix from the model predictions.

    Parameters:
        predictions [np.ndarray]: the models predictions on a given dataset.
        labels [np.ndarray]: the true labels corresponding to the model predictions.
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

def plot_roc_curve(probabilities: np.ndarray, labels: np.ndarray) -> None:
    """
    Plot the ROC curve from the model probabilities.

    Parameters:
        probabilities [np.ndarray]: the model probabilities of guessing each class.
        labels [np.ndarray]: the true labels corresponding to the probabilities.
    """

    fpr, tpr, _ = roc_curve(labels, probabilities)
    score = roc_auc_score(labels, probabilities)

    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.grid()
    plt.legend()
    plt.gca().set_facecolor(mocha_colours.mantle.hex)

    outpath = os.path.join(plots_dir, "roc_curve.png")
    plt.savefig(outpath)
    
    print(f"Saved {outpath}")
    plt.close()

    return None
