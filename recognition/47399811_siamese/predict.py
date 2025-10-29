"""

"""


import numpy as np
import torch
from torch.utils.data import DataLoader 
from pytorch_grad_cam import GradCAM
from sklearn.metrics import roc_auc_score

from modules import SiameseNetwork, Classifier, load_model
from dataset import ISICImageDataset, get_train_validation_test_dataloaders
from plotting import plot_image_showcase


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu': print("Warning using CPU!")

def grad_cam(siamese: SiameseNetwork, images: list, labels: list):
    target_layers = [siamese.final_convolution_layer]

    with GradCAM(model=siamese, target_layers=target_layers) as cam:
        cams = cam(images, targets=None)
    
    images = [np.transpose(img.cpu().numpy(), axes=(1,2,0)) for img in images]
    # conver images to be in the range of 0 to 1
    images = [(img + np.abs(np.min(img)))/(np.max(img) + np.abs(np.min(img))) for img in images]

    plot_image_showcase(images, labels, "grad_cam_image_showcase", "GRAD-CAM Images", cams)

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
            classifier_out = classifier(latent_vector)
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
    siamese = load_model('siamese').to(device)
    classifer = load_model('classifier').to(device)

    image_dir = './data/images/'
    labels_filepath = './data/ISIC_2020_Training_GroundTruth.csv'

    _, _, test_dataloader = get_train_validation_test_dataloaders()
    test_accuracy(siamese, classifer, test_dataloader)

    dataset = ISICImageDataset(image_dir, labels_filepath)
    dataloader = dataset.to_DataLoader(batch_size=9)
    dataloader.dataset.set_triple_iter(False)
    
    for images, labels in dataloader:
        images.to(device)
        grad_cam(siamese, images, labels.cpu().numpy())
        
        latent_vector = siamese(images)
        classifier_out = classifer(latent_vector)

        break

    probs = torch.softmax(classifier_out, dim=1)
    preds = torch.argmax(classifier_out, dim=1)

    captions = list()
    for i in range(9):
        pred = "Malignant" if preds[i] else "Benign"
        label = "Malignant" if labels[i] else "Benign"
        caption = f"Model: pred: {pred} with probability {(100*probs[i][preds[i]]):.1f}%\nLabel: {label}" 
        captions.append(caption)

    plot_image_showcase(images, labels, 'predictions_image_showcase.png', "Image Predictions", captions=captions)