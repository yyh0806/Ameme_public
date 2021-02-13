import torch.nn as nn
import torch.nn.functional as F

from base import ModelBase
from logger.logger import setup_logging
import torchvision
from efficientnet_pytorch import EfficientNet
from .models.transformer import ViT
import timm


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


class EfficientB0Model(ModelBase):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientB4Model(ModelBase):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientB6Model(ModelBase):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class Resnext50_32x4d(ModelBase):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = torchvision.models.resnext50_32x4d(num_classes=num_classes, pretrained=False)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class EffB4NS(ModelBase):
    def __init__(self, num_classes):
        super(EffB4NS, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b4_ns', pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, num_classes)
    def forward(self, x):
        x = self.model(x)
        return x


class ViTBase16(ModelBase):
    def __init__(self, num_classes):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model('vit_base_patch16_384', pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class Transformer(ModelBase):
    def __init__(self, image_size, patch_size, num_classes, channels, dim, depth, heads, mlp_dim, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, channels=channels,
                         dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout).cuda()

    def forward(self, x):
        x = self.model(x)
        return x
