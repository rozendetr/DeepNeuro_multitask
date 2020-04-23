from .resnet import *
import torch.nn as nn
import torch

model_resnet = resnet18()
expansion = 1


class ResNet_2head(nn.Module):
    def __init__(self, expansion=1, num_classes_1=10, num_classes_2=10):
        super(ResNet_2head, self).__init__()
        self.expansion = expansion
        self.share_model = nn.Sequential(*list(model_resnet.children())[:-1])

        # nn.Sequential(*list(model_resnet.children())[-1]).in
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(512 * self.expansion, 10 * self.expansion)
        self.bn1 = nn.BatchNorm1d(10 * self.expansion, eps=2e-1)

        #heads
        self.h1 = nn.Linear(10 * self.expansion, num_classes_1)
        self.h2 = nn.Linear(10 * self.expansion, num_classes_2)

    def forward(self, x):
        x = self.share_model(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)

        y1 = self.h1(x)
        y2 = self.h2(x)

        return y1, y2

    def freeze_shared(self, check_freeze=False):
        for param in self.share_model.parameters():
            param.requires_grad = check_freeze






