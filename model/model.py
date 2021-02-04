import torch.nn as nn
import torch.nn.functional as F

from base import ModelBase
from logger.logger import setup_logging
import torchvision
from efficientnet_pytorch import EfficientNet


class MnistModel(ModelBase):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# class CassavaModel(ModelBase):
#     def __init__(self, num_classes=5):
#         super().__init__()
#         self.model = torchvision.models.resnet152()
#         self.model.fc = nn.Linear(2048, num_classes, bias=True)
#
#     def forward(self, x):
#         x = self.model(x)
#         return x


class EfficientB4Model(ModelBase):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class Resnet50(ModelBase):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.model = torchvision.models.resnet50()