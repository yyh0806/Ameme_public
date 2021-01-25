import abc

import torchvision.transforms as T


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

    MEANS = (0.485, 0.456, 0.406)
    STDS = (0.229, 0.224, 0.225)

    def build_train(self):
        return T.Compose(
            [
             T.Resize((256, 256)),
             T.RandomRotation(90),
             T.RandomHorizontalFlip(p=0.5),
             T.ToTensor(),
             T.Normalize(self.MEANS, self.STDS)])

    def build_test(self):
        return T.Compose(
            [
             T.Resize((256, 256)),
             T.RandomRotation(90),
             T.RandomHorizontalFlip(p=0.5),
             T.ToTensor(),
             T.Normalize(self.MEANS, self.STDS)])