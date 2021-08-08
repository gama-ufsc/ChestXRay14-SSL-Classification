import cv2.cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
import albumentations as A
from albumentations.pytorch import ToTensorV2

# https://pytorch.org/vision/stable/transforms.html

original_torchvision = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15,
                            translate=(0.05, 0.05),
                            scale=(0.95, 1.05)
                            ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


strong_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=25,
                            translate=(0.15, 0.15),
                            scale=(0.9, 1.1)
                            ),
    transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

medium_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15,
                            translate=(0.1, 0.1),
                            scale=(0.9, 1.1)
                            ),
    transforms.RandomPerspective(distortion_scale=0.1, p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


random_crop = transforms.Compose([
    transforms.Resize(300),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15,
                            translate=(0.05, 0.05),
                            scale=(0.95, 1.05)
                            ),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

strong_random_crop = transforms.Compose([
    transforms.Resize(300),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=25,
                            translate=(0.2, 0.2),
                            scale=(0.8, 1.2)
                            ),
    transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

five_cropped = transforms.Compose(
    [
        transforms.Resize(304),  # 256+48
        transforms.FiveCrop(256),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),

        transforms.Lambda(lambda crops: torch.stack([transforms.RandomHorizontalFlip()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.RandomAffine(degrees=15,
                                                                             translate=(0.05, 0.05),
                                                                             scale=(0.95, 1.05)
                                                                             )(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
    ])


def train_data_loader(data_dir, image_list, batch_size=16, transforms='original'):
    if transforms == 'original':
        data_transforms = original_torchvision
    elif transforms == 'strong':
        data_transforms = strong_transforms
    elif transforms == 'strong_random_crop':
        data_transforms = strong_random_crop
    elif transforms == 'random_crop':
        data_transforms = random_crop
    elif transforms == 'medium_transforms':
        data_transforms = medium_transforms
    else:
        print("Could not find specified transformations")
        raise
    print(f"=> using a data loader of type [{transforms}]")

    dataset = ChestXrayDataSet(data_dir=data_dir,
                               image_list_file=image_list,
                               transform=data_transforms
                               )

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return loader


def validation_data_loader(data_dir, image_list, batch_size=16):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda
        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])

    dataset = ChestXrayDataSet(data_dir=data_dir,
                               image_list_file=image_list,
                               transform=transform
                               )

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return loader
