# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # 5×5卷积核，无填充
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 第一层卷积+池化：24×24→12×12
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 第二层卷积+池化：8×8→4×4
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # 展平至1024维
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
