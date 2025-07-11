import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p=0.1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.activation2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(p=0.1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        
        self.flattened_size = self._get_flattened_size(input_shape)
        self.fc = nn.Linear(self.flattened_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _get_flattened_size(self, input_shape):
        x = torch.randn(1, *input_shape)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.dropout3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        return x.size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.dropout3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
