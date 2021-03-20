from model.models import create_model
from base import ModelBase
import torch.nn.functional as F
import torch.nn as nn


class MnistModel(ModelBase):
    def __init__(self, num_classes: int):
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


class Efficient_b4(ModelBase):
    def __init__(self, num_classes: int):
        """

        :param num_classes: int
        """
        super(Efficient_b4, self).__init__()
        self.model = create_model('tf_efficientnet_b4_ns', num_classes=num_classes, pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x


class Efficient_b6(ModelBase):
    def __init__(self, num_classes: int):
        """

        :param num_classes: int
        """
        super(Efficient_b6, self).__init__()
        self.model = create_model('tf_efficientnet_b6_ns', num_classes=num_classes, pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet50D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = create_model('resnet50d', pretrained=False)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output


class ResNet200D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = create_model('resnet200d', pretrained=False)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output


class ViTBase16(ModelBase):
    def __init__(self, num_classes: int):
        """

        :param num_classes:
        """
        super(ViTBase16, self).__init__()
        self.model = create_model('vit_base_patch16_384', num_classes=num_classes, pretrained=False)

    def forward(self, x):
        x = self.model(x)
        return x
