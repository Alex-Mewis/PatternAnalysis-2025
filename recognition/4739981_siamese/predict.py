"""
contains code showing usage of the trained model.
Prints out any results and provided visualsations.
"""
import os
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns



#### PERAMBLE #####################################################################
this_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(this_dir, "plots")

if not os.path.exists(plots_dir): os.mkdir(plots_dir)


#### PLOTTING FUNCTIONS ###########################################################
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
    return None