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
from torchvision import transforms

PERCENTAGE_OF_DATA_TO_LOAD = 0.1 
THREADS_USE = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

to_tesnor_transform = transforms.ToTensor()

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

def fetch_image_data(filepath: str) -> None:
    image = Image.open(filepath)
    return to_tesnor_transform(image).to(device)

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
                 shortcut_images: np.ndarray | None = None,
                 shortcut_labels: np.ndarray | None = None) -> None:

        if shortcut_images is not None and shortcut_labels is not None:
            assert len(shortcut_labels) == len(shortcut_images)
            self._len = len(shortcut_labels)
            self._image_filepaths = shortcut_images
            self._labels = shortcut_labels
            return None

        # get the image data.
        self._len = int(len(os.listdir(image_dir)) * PERCENTAGE_OF_DATA_TO_LOAD)
        
        labels_df = pd.read_csv(labels_path)
        self._labels = np.zeros(self._len)
        loading_args = [None] * self._len
        self._image_filepaths = np.empty(self._len, dtype=object)
        number_of_images_loaded = [0]
        
        for i, image_name in enumerate(os.listdir(image_dir)):
            if i >= self._len: break
            loading_args[i] = [i, os.path.join(image_dir, image_name), self._image_filepaths, 
                               labels_df, self._labels, number_of_images_loaded]

        launch_progress_bar(number_of_images_loaded, self._len)
        
        with ThreadPoolExecutor(max_workers=THREADS_USE) as thread_executer:
            for args in loading_args:
                thread_executer.submit(load_image, args)
        
        self._triple_iter = True
        
        return None
    
    def __len__(self) -> int:
        return self._len
    
    def __find_random_pair(self, i: int, label: int) -> torch.Tensor:
        """
        """
        found_pair = False
        while not found_pair:
            pair_i = random.randint(0, self._len-1)
            if pair_i == i: continue
            pair_label = self._labels[pair_i]
            found_pair = pair_label == label
        
        return fetch_image_data(self._image_filepaths[pair_i])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int] | tuple[torch.Tensor, int]:
        if (i < 0 or i >= self._len):
            raise IndexError
        
        image = fetch_image_data(self._image_filepaths[i])
        label = self._labels[i]

        if not self._triple_iter:
            return image, label

        positive_image = self.__find_random_pair(i, label)
        negative_image = self.__find_random_pair(i, not label)
        
        return image, positive_image, negative_image, label
    
    def set_triple_iter(self, triple_iter: bool) -> None:
        self._triple_iter = triple_iter 
        return None

    def shuffle(self) -> None:
        """
        """
        shuffled_indicies = list(range(self._len)) 
        random.shuffle(shuffled_indicies)
        
        self._image_filepaths = self._image_filepaths[shuffled_indicies]
        self._labels = self._labels[shuffled_indicies]
        
        return None

    def split(self, p: float = 0.8, shuffle: bool = True) -> tuple[Self, Self]:
        """
        """
        if shuffle: self.shuffle()

        train_n = int(p*self._len)

        train_image_filepaths = self._image_filepaths[:train_n]
        test_image_filepaths = self._image_filepaths[train_n:]
        
        train_labels = self._labels[:train_n]
        test_labels = self._labels[train_n:]
        
        train_dataset = ISICImageDataset(None, None, train_image_filepaths, train_labels)
        test_dataset = ISICImageDataset(None, None, test_image_filepaths, test_labels)

        return train_dataset, test_dataset

    def to_DataLoader(self, **kwargs) -> DataLoader:
        return DataLoader(self, **kwargs)
    


if __name__ == "__main__":
    get_data_stats('./data/ISIC_2020_Training_GroundTruth.csv')