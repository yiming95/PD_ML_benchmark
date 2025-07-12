import torch.nn as nn
import torch

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        # 定义各层
        self.fc0 = nn.Linear(512, 512)
        self.bn0 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        self.fc7 = nn.Linear(32, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.fc8 = nn.Linear(32, 16)
        self.bn8 = nn.BatchNorm1d(16)
        self.fc9 = nn.Linear(16, 16)
        self.bn9 = nn.BatchNorm1d(16)
        self.fc10 = nn.Linear(16, 8)
        self.bn10 = nn.BatchNorm1d(8)
        self.fc11 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 应用ReLU激活函数和前向传播，每个全连接层后加入批量归一化层
        x = self.relu(self.bn0(self.fc0(x)))
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.relu(self.bn6(self.fc6(x)))
        x = self.relu(self.bn7(self.fc7(x)))
        x = self.relu(self.bn8(self.fc8(x)))
        x = self.relu(self.bn9(self.fc9(x)))
        x = self.relu(self.bn10(self.fc10(x)))
        x = self.fc11(x) # 输出层使用Sigmoid激活函数
        return x
