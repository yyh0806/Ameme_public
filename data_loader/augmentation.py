import abc
import cv2
import torchvision.transforms as T
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, Cutout,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2


class AugmentationFactoryBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class MNISTTransforms(AugmentationFactoryBase):
    MEANS = [0]
    STDS = [1]

    def build_train(self):
        return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])

    def build_test(self):
        return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])


class CassavaTransforms(AugmentationFactoryBase):
    def build_train(self):
        train_transform = Compose([
            RandomResizedCrop(width=512, height=512),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0)
        ], p=1)
        return train_transform

    def build_test(self):
        test_transform = Compose([
            CenterCrop(512, 512, p=1.0),
            Resize(512, 512, p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)
        ], p=1.0)
        return test_transform
