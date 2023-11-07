import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)  # 20个5x5的卷积核，输入通道数为1
        self.pool = nn.MaxPool2d(2, 2)    # 2x2的最大池化
        self.conv2 = nn.Conv2d(20, 50, 5) # 50个5x5的卷积核
        self.fc1 = nn.Linear(50 * 4 * 4, 500)  # 全连接层
        self.fc2 = nn.Linear(500, 10)         # 输出层

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 50 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return 0, x
