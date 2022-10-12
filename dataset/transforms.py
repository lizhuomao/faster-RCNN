import random

from torchvision.transforms import functional as F


class Compose(object):
    """
        compose multiple transform function
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for ts in self.transforms:
            image, target = ts(image, target)

        return image, target


class ToTensor(object):
    """
        made PIL image to Tensor type
    """

    def __call__(self, image, target):
        image = F.to_tensor()
        return image, target


class RandomHorizontalFlip(object):
    """
        randomly horizontal flip image and corresponding bboxes
    """
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bboxes = target["bboxes"]
            bboxes[: [0, 2]] = width - bboxes[:, [2, 0]]
            target["bboxes"] = bboxes
            target["bboxes"] = bboxes
        return image, target
