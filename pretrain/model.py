import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResModel(nn.Module):
    def __init__(self):
        super(ResModel, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        self.out_layer = nn.Linear(1000, 4)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.out_layer(x)
        return F.log_softmax(x, dim=1)
