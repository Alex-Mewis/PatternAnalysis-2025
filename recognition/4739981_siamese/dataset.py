"""
Contains the data loader for loading and preprocesing the data.
"""
import os, time, random, threading
from typing import Self
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PERCENTAGE_OF_DATA_TO_LOAD = 0.1 
THREADS_USE = 4 

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

def launch_progress_bar(n, total: int):
    
    def update_progress_bar() -> None:
        print("Started Loading in Data")
        start_time = time.time()
        while True:
            progress = n[0]/total
            progress_bar_animation(progress)
            if progress == 1:
                break
            time.sleep(0.1)

        elapsed_time = time.time() - start_time
        print(f"\nLoading in Images took {elapsed_time:.3f}s or {(elapsed_time/60):3f}mins")
        return None
    
    thread = threading.Thread(target=update_progress_bar)
    thread.daemon = True
    thread.start()
    return

def load_image_data(args: list) -> None:
    image_path, image_tensors, i, n = args
    image = Image.open(image_path)
    image_tensors[i] =  to_tesnor_transform(image)
    n[0] += 1
    return None

class ISICImageDataset(Dataset):
    """
    """
    def __init__(self, image_dir: str, labels_path: str,
                 shortcut_images: torch.Tensor | None = None,
                 shortcut_labels: np.ndarray | None = None) -> None:

        if shortcut_images is not None and shortcut_labels is not None:
            assert len(shortcut_labels) == len(shortcut_images)
            self._len = len(shortcut_labels)
            self._images_data = shortcut_images
            self._labels = shortcut_labels
            return None

        # get the image data.
        self._len = int(len(os.listdir(image_dir)) * PERCENTAGE_OF_DATA_TO_LOAD)
        image_tensors = [None] * self._len 
        image_names = [None] * self._len
        image_args = [None] * self._len
        number_of_images_loaded = [0] 
        for i, image_name in enumerate(os.listdir(image_dir)):
            if i >= self._len: break
            image_names[i] = image_name.replace('.jpg', '')
            image_args[i] = [os.path.join(image_dir, image_name), image_tensors, i, number_of_images_loaded]

        launch_progress_bar(number_of_images_loaded, self._len)

        with ThreadPoolExecutor(max_workers=THREADS_USE) as thread_excecuter:
            for args in image_args:
                thread_excecuter.submit(load_image_data, args)
        
        self._images_data = torch.stack(image_tensors, dim=0)

        # get the labels data.
        labels_df = pd.read_csv(labels_path)
        self._labels = np.zeros(self._len)
        for i, image_name in enumerate(image_names):
            label = labels_df.loc[labels_df['image_name'] == image_name, 'target'].iloc[0]
            self._labels[i] = label

        self._iter_pairwise = True
        
        return None
    
    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int] | tuple[torch.Tensor, int]:
        if (i < 0 or i >= self._len):
            raise IndexError
        
        image = self._images_data[i]
        label = self._labels[i]

        if not self._iter_pairwise:
            return image, label

        # need to find a matching image to pair up with.
        # this matching image needs to be found at random.
        found_pair = False
        while not found_pair:
            pair_i = random.randint(0, self._len-1)
            if pair_i == i: continue
            pair_label = self._labels[pair_i]
            found_pair = pair_label == label

        pair_image = self._images_data[pair_i]
        return image, pair_image, label
    
    def set_iter_pairwise(self, iter_pairwise: bool) -> None:
        self._iter_pairwise = iter_pairwise
        return None

    def shuffle(self) -> None:
        """
        """
        shuffled_indicies = list(range(self._len)) 
        random.shuffle(shuffled_indicies)
        
        self._images_data = self._images_data[shuffled_indicies]
        self._labels = self._labels[shuffled_indicies]
        
        return None

    def split(self, p: float = 0.8, shuffle: bool = True) -> tuple[Self, Self]:
        """
        """
        if shuffle: self.shuffle()

        train_n = int(p*self._len)

        train_images = self._images_data[:train_n]
        test_images = self._images_data[train_n:]
        
        train_labels = self._labels[:train_n]
        test_labels = self._labels[train_n:]
        
        train_dataset = ISICImageDataset(None, None, train_images, train_labels)
        test_dataset = ISICImageDataset(None, None, test_images, test_labels)

        return train_dataset, test_dataset

    def to_DataLoader(self, **kwargs) -> DataLoader:
        return DataLoader(self, **kwargs)