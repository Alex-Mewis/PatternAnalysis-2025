"""
Contains all of the functions used to evaluate and use/predict with the trained model.

Made by: Alexander Mewis
"""
import numpy as np
import torch
from torch.utils.data import DataLoader 
from pytorch_grad_cam import GradCAM
from sklearn.metrics import roc_auc_score

from modules import SiameseNetwork, load_model
from dataset import ISICImageDataset, get_train_validation_test_dataloaders
from plotting import plot_image_showcase, plot_confusion_matrix, plot_roc_curve


#### PERAMBLE #####################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu': print("Warning using CPU!")


#### FUNCTIONS ####################################################################
def grad_cam(model: SiameseNetwork, images: list, labels: list) -> None:
    """
    Runs the model on a small batch of images and record the cams gradient using
    GrandCAM. Then make an image showcase plot with the images and the GradCam
    overlay.

    Parameters:
        model [SiameseNetwork]: the model whose grad-cam output is being made.
        images [list]: a list of images where the model will get a grad-cam output for.
        labels [list]: a list of true labels corresponding to the images.
    """
    target_layers = [model.final_convolution_layer]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        cams = cam(images, targets=None)
    
    images = [np.transpose(img.cpu().numpy(), axes=(1,2,0)) for img in images]
    # conver images to be in the range of 0 to 1
    images = [(img + np.abs(np.min(img)))/(np.max(img) + np.abs(np.min(img))) for img in images]

    plot_image_showcase(images, labels, "grad_cam_image_showcase", "GRAD-CAM Images", cams)

    return None

def test_accuracy(model: SiameseNetwork, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tests the models accuracy on the test dataset and generates the model probabilities and
    predictions the test dataset.
    Prints out the following statistics on the test dataset:
        - Accuracy
        - AUC-ROC score
    
    Parameters:
        model [SiameseNetwork]: the model whose accuarcy needs to be tested.
        test_loader [DataLoader]: the datalaoder for the test dataset.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (the labels of test dataset, 
            the model probabilities fo each class guess, the model predictions for each image).
    """
    test_loader.dataset.set_triple_iter(False)

    model.eval()

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

            classifier_out = model.classify(imgs)
            probs = torch.softmax(classifier_out, dim=1)[:, 1]
            preds = torch.argmax(classifier_out, dim=1)

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


if __name__ == "__main__":
    model = load_model().to(device)

    _, _, test_dataloader = get_train_validation_test_dataloaders()

    # test the model accuracy against the test data
    labels, probabilities, predictions = test_accuracy(model, test_dataloader)
    plot_confusion_matrix(predictions, labels)
    plot_roc_curve(probabilities, labels)

    # make the gradcam images
    test_dataloader.dataset.set_triple_iter(False)
    for images, labels in test_dataloader:
        images, labels = images[:9], labels[:9]
        images.to(device)
        grad_cam(model, images, labels.cpu().numpy())
        classifier_out = model.classify(images)
        break

    probs = torch.softmax(classifier_out, dim=1)
    preds = torch.argmax(classifier_out, dim=1)

    # plot model image predictions showcase.
    captions = list()
    for i in range(9):
        pred = "Malignant" if preds[i] else "Benign"
        label = "Malignant" if labels[i] else "Benign"
        caption = f"Model: pred: {pred}\nwith probability {(100*probs[i][preds[i]]):.1f}%\nLabel: {label}" 
        captions.append((caption, pred==label))

    images = np.transpose(images.cpu().numpy(), axes=(0, 2, 3, 1))
    images = [(img + np.abs(np.min(img)))/(np.max(img) + np.abs(np.min(img))) for img in images]

    plot_image_showcase(images, labels, 'predictions_image_showcase', "Image Predictions", captions=captions)