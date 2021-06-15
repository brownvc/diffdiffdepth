import torch
import torch.nn as nn
from torchvision import models

vgg16 = models.vgg16(pretrained=True)

class VggFeatures(nn.Module):
    def __init__(self):
        super(VggFeatures, self).__init__()
        self.features = nn.Sequential(*list(vgg16.features.children())[:4]) #[:4]

    def forward(self, x):
        x = self.features(x)
        return x
