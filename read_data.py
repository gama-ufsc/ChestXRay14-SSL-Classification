# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None, return_image_name=False):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.return_img_name = return_image_name
        if 'albumentations' in str(transform.__class__):
            self.transform_type = 'albumentations'
        elif 'torchvision' in str(transform.__class__):
            self.transform_type = 'torchvision'
        print(f"Augmentation type : {self.transform_type} --- from [{str(transform.__class__)}]")
    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]

        if self.transform_type == 'torchvision':
            image = Image.open(image_name).convert('RGB')
            label = self.labels[index]
            if self.transform is not None:
                image = self.transform(image)
        elif self.transform_type == 'albumentations':
            image = np.array(Image.open(image_name).convert('RGB'))
            label = self.labels[index]
            if self.transform is not None:
                image = self.transform(image=image)
            image = image['image']
        if self.return_img_name:
            return image_name, image, torch.FloatTensor(label)
        else:
            return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
