import torch.nn as nn
from base import ModelBase
from .models import create_model


class Resnext50_32x4d(ModelBase):
    def __init__(self, num_classes, pretrained):
        super(Resnext50_32x4d, self).__init__()
        self.model = create_model('resnext50_32x4d', pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.classifier = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientB4(ModelBase):
    def __init__(self, num_classes, pretrained):
        super(EfficientB4, self).__init__()
        self.model = create_model('tf_efficientnet_b4_ns', pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class ViTBase16(ModelBase):
    def __init__(self, num_classes, pretrained):
        super(ViTBase16, self).__init__()
        self.model = create_model('vit_base_patch16_384', pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
