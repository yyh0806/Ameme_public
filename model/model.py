from model.models import create_model
from base import ModelBase


class Efficient_b4(ModelBase):
    def __init__(self, num_classes):
        super(Efficient_b4, self).__init__()
        self.model = create_model('tf_efficientnet_b4_ns', num_classes=num_classes, pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x


class ViTBase16(ModelBase):
    def __init__(self, num_classes):
        super(ViTBase16, self).__init__()
        self.model = create_model('vit_base_patch16_384', num_classes=num_classes, pretrained=False)

    def forward(self, x):
        x = self.model(x)
        return x
