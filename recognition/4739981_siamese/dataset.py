"""
Contains the data loader for loading and preprocesing the data.
"""
import os, random
from typing import Self
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PERCENTAGE_OF_DATA_TO_LOAD = 1

to_tesnor_transform = transforms.ToTensor()


class ISICImageDataset(Dataset):
    """
    """
    def __init__(self, image_dir: str, labels_path: str,
                 shortcut_images: torch.Tensor | None = None,
                 shortcut_labels: np.ndarray | None = None) -> None:

        if shortcut_images is not None and shortcut_labels is not None:
            self._len = len(shortcut_labels)
            self._iamges_data = shortcut_images
            self._labels = shortcut_labels
            return None

        # get the image data.
        self._len = int(len(os.listdir(image_dir)) * PERCENTAGE_OF_DATA_TO_LOAD)
        image_tensors = [None] * self._len 
        image_names = [None] * self._len
        for i, image_name in enumerate(os.listdir(image_dir)):
            if i >= self._len: break
            image = Image.open(os.path.join(image_dir, image_name))
            image_tensors[i] =  to_tesnor_transform(image)
            image_names[i] = image_name.replace(".jpeg", "")

        self._images_data = torch.stack(image_tensors, dim=0)

        # get the labels data.
        labels_df = pd.read_csv(labels_path)
        self._labels = np.zeros(self._len)
        for i, image_name in enumerate(image_names):
            label = labels_df.loc[labels_df['image_name'] == image_name, 'target'].iloc[0]
            self._labels[i] = label 
        
        return None
    
    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if (idx < 0 or idx >= self._len):
            raise IndexError
        image = self._images_data[idx]
        label = self._labels[idx]
        return image, label

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
        test_labels = self._labels[:train_n]
        
        train_dataset = ISICImageDataset(None, None, train_images, train_labels)
        test_dataset = ISICImageDataset(None, None, test_images, test_labels)

        return train_dataset, test_dataset

    def to_DataLoader(self, **kwargs) -> DataLoader:
        return DataLoader(self, **kwargs)