
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from modules import SiameseNetwork
from train import load_model
from dataset import ISICImageDataset
from plotting import plot_image_showcase


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu': print("Warning using CPU!")

def grad_cam(siamese: SiameseNetwork, images: list):
    target_layers = [siamese.final_layer]

    with GradCAM(model=siamese, target_layers=target_layers) as cam:
        cams = cam(images, targets=None)
    
    images = [np.transpose(img.cpu().numpy(), axes=(1,2,0)) for img in images]
    # conver images to be in the range of 0 to 1
    images = [(img + np.abs(np.min(img)))/(np.max(img) + np.abs(np.min(img))) for img in images]

    plot_image_showcase(images, "grad_cam_image_showcase", "GRAD-CAM Images", cams)



if __name__ == "__main__":
    siamese = load_model('siamese').to(device)
    
    image_dir = './data/images/'
    labels_filepath = './data/ISIC_2020_Training_GroundTruth.csv'


    dataset = ISICImageDataset(image_dir, labels_filepath)
    dataloader = dataset.to_DataLoader(batch_size=9)
    dataloader.dataset.set_triple_iter(False)
    
    for images, labels in dataloader:
        images.to(device)
        siamese(images)
        grad_cam(siamese, images)
        break

