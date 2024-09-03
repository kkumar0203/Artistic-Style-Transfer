"""
Ref = https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/utils.py
"""
from collections import namedtuple

import torch.nn as nn
from torchvision import models


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):  # relu1_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):  # relu2_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):  # relu3_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):  # relu4_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(x)
        h_relu2_2 = h
        h = self.slice3(x)
        h_relu3_3 = h
        h = self.slice4(x)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
