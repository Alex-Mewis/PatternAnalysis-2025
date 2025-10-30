"""
Contains the data loader for loading and preprocesing the data.
"""
import os, time, random, threading
from typing import Self
from concurrent.futures import ThreadPoolExecutor 
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from plotting import plot_image_showcase
from configs import BATCH_SIZE 

PERCENTAGE_OF_DATA_TO_LOAD = 1.0 
THREADS_USE = 4

random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#### CONFIGURABLES ###########################################################
TRAIN_IMAGE_TRANSFORMS = v2.Compose([
    v2.RandomRotation(10),
    v2.RandomVerticalFlip(),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VALIDATION_TRANSFORMS = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def progress_bar_animation(progress: float, num_boxes:int = 66, completed_symbol:str = '∎', waiting_symbol:str = '□') -> None:
    """ 
    print out a loading bar animation given the progress made

    Paramters:
        progress [float] : the percentage of the tast which has been completed
            is a float usually a float between 0 and 1
        num_boxes [int] : the number of boxes in the loading animation
    """
    if 0 <= progress:
        num_filled_boxes = int(progress*num_boxes)
        loading_boxes = completed_symbol*num_filled_boxes  + waiting_symbol*(num_boxes - num_filled_boxes) if not num_filled_boxes > num_boxes else completed_symbol*num_boxes
        loading_bar = "["+loading_boxes+"]"
        percentage = '%g'%round(progress*100, 0) if progress <= 1 else "100"
        while len(percentage) < 3:
            percentage = " " + percentage
        print(f"Loading {percentage}% : " + loading_bar, end="\r")
    
    return None

def launch_progress_bar(n, total: int) -> None:

    def update_progress_bar() -> None:
        print("Started Loading in images")
        start_time = time.time()
        while True:
            progress = n[0]/total
            progress_bar_animation(progress)
            if progress == 1: break
            time.sleep(0.1)
        
        elapsed_time = time.time() - start_time
        print(f"\nLoading in Images took {elapsed_time:.3f}s or {(elapsed_time/60):.3f}mins")
        return None

    thread = threading.Thread(target=update_progress_bar)
    thread.daemon = True
    thread.start()

    return None 

def load_image(args: list) -> None: 
    i, image_filepath, image_filepaths, labels_df, labels, n = args
    
    image_filepaths[i] = image_filepath 

    image_name = os.path.basename(image_filepath).replace('.jpg', '')        
    label = labels_df.loc[labels_df['image_name'] == image_name, 'target'].iloc[0]
    labels[i] = label

    n[0] += 1

    return None

def get_data_stats(labels_path: str) -> None:
    """
    """
    df = pd.read_csv(labels_path)
    n = len(df)
    n_benign = (df['target'] == 0).sum()
    n_malignant = (df['target'] == 1).sum()

    print(f"Number of datapoints = {n}")
    print(f"Number of benign     = {n_benign} => {(100*n_benign/n):.3f}%")
    print(f"Number of maligant   = {n_malignant}   => {(100*n_malignant/n):.3f}%")

    return None

class ISICImageDataset(Dataset):
    """
    """
    def __init__(self, image_dir: str, labels_path: str,
                 shortcut_positive_images: np.ndarray | None = None,
                 shortcut_negative_images: np.ndarray | None = None) -> None:

        self._transforms =  VALIDATION_TRANSFORMS

        if shortcut_positive_images is not None and shortcut_negative_images is not None:
            self._positive_images = shortcut_positive_images
            self._negative_images = shortcut_negative_images
            return None

        # get the image data.
        n = int(len(os.listdir(image_dir)) * PERCENTAGE_OF_DATA_TO_LOAD)
        
        labels_df = pd.read_csv(labels_path)
        labels = np.zeros(n)
        loading_args = [None] * n 
        image_filepaths = [None] * n 
        number_of_images_loaded = [0]
        
        for i, image_name in enumerate(os.listdir(image_dir)):
            if i >= n: break
            loading_args[i] = [i, os.path.join(image_dir, image_name), image_filepaths, 
                               labels_df, labels, number_of_images_loaded]

        launch_progress_bar(number_of_images_loaded, n)
        
        with ThreadPoolExecutor(max_workers=THREADS_USE) as thread_executer:
            for args in loading_args:
                thread_executer.submit(load_image, args)

        # sort the images into positive and negative labels.
        self._positive_images = list()
        self._negative_images = list()
        for label in labels:
            if not label:
                self._negative_images.append(image_filepaths.pop(0))
            else:
                self._positive_images.append(image_filepaths.pop(0))
        
        random.shuffle(self._negative_images)
        random.shuffle(self._positive_images)
        
        self._triple_iter = True
        
        return None
    
    def __len__(self) -> int:
        return len(self._negative_images) + len(self._positive_images) 
    
    def set_training_data_transforms(self) -> None:
        self._transforms = TRAIN_IMAGE_TRANSFORMS
        return None
    
    def set_transforms(self, transforms) -> None:
        self._transforms = transforms 
        return None

    def force_even_data(self) -> None:

        # increase the positive images to be the same size as the negative
        n = len(self._positive_images)
        pos_i = 0
        for i in range(len(self._positive_images), len(self._negative_images)):
            self._positive_images.append(self._positive_images[pos_i])
            pos_i = pos_i + 1 if pos_i != n - 1 else 0

        random.shuffle(self._positive_images)

        # TODO remove this check
        assert len(self._positive_images) == len(self._negative_images) 

        return None

    def __fetch_image_data(self, filepath: str) -> torch.Tensor:
        image = Image.open(filepath).convert("RGB")
        image_data = self._transforms(image) 
        return image_data.to(device)

    def __find_random_pair(self, i: int, label: int, image_filepath: str) -> torch.Tensor:
        """
        """
        search_space = self._positive_images if label else self._negative_images
        pair_i = i

        while pair_i == i or search_space[pair_i] == image_filepath:
            pair_i = random.randint(0, len(search_space)-1)

        return self.__fetch_image_data(search_space[pair_i])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, float] | tuple[torch.Tensor, float]:
        if (i < 0 or i >= len(self)):
            raise IndexError

        if i >= 2*len(self._positive_images) + 1:
            filepath = self._negative_images[i - len(self._positive_images)]
            label = 0.0
        elif i % 2:
            filepath = self._positive_images[(i-1)//2]
            label = 1.0
        else:
            filepath = self._negative_images[(i-1)//2]
            label = 0.0

        image = self.__fetch_image_data(filepath)

        if not self._triple_iter:
            return image, label

        positive_image = self.__find_random_pair(i, label, filepath)
        negative_image = self.__find_random_pair(i, not label, filepath)
        
        return image, positive_image, negative_image, label
    
    def set_triple_iter(self, triple_iter: bool) -> None:
        self._triple_iter = triple_iter 
        return None

    def split(self, p: float = 0.8) -> tuple[Self, Self]:
        """
        """

        n_pos = int(p*len(self._positive_images))
        n_neg = int(p*len(self._negative_images))

        train_positive_filepaths = self._positive_images[:n_pos]
        test_positive_filepaths = self._positive_images[n_pos:]
        train_negative_filepaths = self._negative_images[:n_neg]
        test_negative_filepaths = self._negative_images[n_neg:]


        train_dataset = ISICImageDataset(None, None, train_positive_filepaths, train_negative_filepaths)
        test_dataset = ISICImageDataset(None, None, test_positive_filepaths, test_negative_filepaths)

        return train_dataset, test_dataset

    def to_DataLoader(self, **kwargs) -> DataLoader:
        return DataLoader(self, **kwargs)


def get_train_validation_test_dataloaders(images_dir: str | None = None, labels_filepath: str | None = None) -> tuple[DataLoader, DataLoader, DataLoader]:

    if images_dir is None: images_dir = "./data/images/" 
    if labels_filepath is None: labels_filepath = './data/ISIC_2020_Training_GroundTruth.csv'


    dataset = ISICImageDataset(images_dir, labels_filepath)

    train_dataset, hidden_dataset = dataset.split(p=0.7)   
    validation_dataset, test_dataset = hidden_dataset.split(p=0.5)
    
    train_dataset.force_even_data()
    train_dataset.set_training_data_transforms()
    # validation_dataset.force_even_data()
    
    train_dataloader = train_dataset.to_DataLoader(batch_size=BATCH_SIZE)
    validation_dataloader = validation_dataset.to_DataLoader(batch_size=BATCH_SIZE)
    test_dataloader = test_dataset.to_DataLoader(batch_size=BATCH_SIZE)

    return train_dataloader, validation_dataloader, test_dataloader


if __name__ == "__main__":
    image_dir = './data/images/'
    labels_filepath = './data/ISIC_2020_Training_GroundTruth.csv'

    get_data_stats(labels_filepath)

    dataset = ISICImageDataset(image_dir, labels_filepath)
    dataset.set_transforms(v2.Compose(VALIDATION_TRANSFORMS.transforms[:-1])) # remove normalisation
    dataset.set_triple_iter(False)
    dataloader = dataset.to_DataLoader(batch_size=9)

    for images, labels in dataloader:
        images = [np.transpose(img.cpu().numpy(), axes=(1,2,0)) for img in images]
        labels = labels.cpu().numpy()
        break
    
    plot_image_showcase(images, labels, 'base_images_showcase', "Base Images")

    dataloader.dataset.set_transforms(v2.Compose(TRAIN_IMAGE_TRANSFORMS.transforms[:-1])) # remove normalisation

    for images, labels in dataloader:
        images = [np.transpose(img.cpu().numpy(), axes=(1,2,0)) for img in images]
        labels = labels.cpu().numpy()
        break
    
    plot_image_showcase(images, labels, 'transformed_images_showcase', "Transformed Images")


