"""

"""


import numpy as np
import torch
from torch.utils.data import DataLoader 
from pytorch_grad_cam import GradCAM
from sklearn.metrics import roc_auc_score

from modules import SiameseNetwork, load_model
from dataset import ISICImageDataset, get_train_validation_test_dataloaders
from plotting import plot_image_showcase, plot_confusion_matrix, plot_roc_curve


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu': print("Warning using CPU!")

def grad_cam(model: SiameseNetwork, images: list, labels: list):
    target_layers = [model.final_convolution_layer]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        cams = cam(images, targets=None)
    
    images = [np.transpose(img.cpu().numpy(), axes=(1,2,0)) for img in images]
    # conver images to be in the range of 0 to 1
    images = [(img + np.abs(np.min(img)))/(np.max(img) + np.abs(np.min(img))) for img in images]

    plot_image_showcase(images, labels, "grad_cam_image_showcase", "GRAD-CAM Images", cams)

    return None

def test_accuracy(model: SiameseNetwork, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:

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