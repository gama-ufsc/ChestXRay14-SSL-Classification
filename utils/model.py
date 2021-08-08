# encoding: utf-8

"""
The main CheXNet model implementation.
"""


import os
import torch
import torch.nn as nn
import torchvision


class Resnet50(nn.Module):
    def __init__(self, out_size):
        super(Resnet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet50(x)
        return x

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x



def create_model(weights_path=None):
    model = DenseNet121(out_size=1).cuda()
    print("=> created model ")
    if weights_path is None:
        print("=> did not load checkpoints")
        return model
    if os.path.isfile(weights_path):
        print("=> loading checkpoint")
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint)
        print("=> loaded checkpoint")
        return model
    else:
        print("=> no checkpoint found")
        raise("Checkpoint does not exist")


