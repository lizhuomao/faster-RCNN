"""
Generalized R-CNN transform
"""
import torchvision.transforms as ts
import numpy as np
import random
import matplotlib.pyplot as plt

from image_list import ImageList
import math
from typing import List, Tuple

import torchvision
import torch
from torch import nn


def resize_bboxes(bboxes, size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """
    perform operation of resize bounding box
    :param bboxes: bounding box
    :param size: original image size
    :param new_size: resized image size
    :return: resized bounding box
    """
    ratios = [
        torch.tensor(s, dtype = torch.float32, device = bboxes.device) /
        torch.tensor(s_orig, dtype = torch.float32, device = bboxes.device)
        for s, s_orig in zip(new_size, size)
    ]
    ratios_height, ratios_width = ratios

    xmin, ymin, xmax, ymax = bboxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim = 1)


def _resize_image(image, min_size, max_size):
    # type: (Tensor, float, float) -> Tensor
    """
    perform operation of resize image
    :param image: image data
    :param min_size: min size
    :param max_size: max size
    :return: resized image
    """
    im_shape = torch.Tensor(image.shape[-2:])
    im_min_size = float(torch.min(im_shape))
    im_max_size = float(torch.max(im_shape))

    # compute scale between image size and transformed size
    scale = min_size / im_min_size
    if im_max_size * scale > max_size:
        scale = max_size / im_max_size
    # using bilinear interpolate resize image
    image = nn.functional.interpolate(
        image[None], scale_factor=scale, mode="bilinear",
        recompute_scale_factor=True, align_corners=False
    )[0]

    return image


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input and target transformation before feeding the data to a GeneralizedRCNN model.
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        """
        initialize class
        :param min_size: range of the minimum length of image
        :param max_size: range of the maximum length of image
        :param image_mean: image mean
        :param image_std: image standard deviation
        """
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets = None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        """
        forward propagation
        :param images: batch images
        :param targets: batch targets
        :return: transformed batch images and targets
        """
        # resize image and bounding bbox
        images = [img for img in images]
        for i in range(len(images)):
            img = images[i]
            target = targets[i] if targets is not None else None

            if img.dim() != 3:
                raise ValueError("Images is expected to be a list of 3d tensors"
                                 " of shape [C, H, W], but got {}".format(img.shape))
            img = self.normalize(img)
            img, target = self.resize(img, target)
            images[i] = img

        # record image shape of resized
        img_shapes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        img_shape_list = torch.jit.annotate(List[Tuple[int, int]], [])
        for img_shape in img_shapes:
            assert len(img_shape) == 2, "Error: image shape is expected to be a length of 2" \
                                        "but get {}".format(len(img_shape))
        img_list = ImageList(images, img_shape_list)
        return img_list, targets

    def normalize(self, image):
        """
        normalize manipulation
        :param image: image data
        :return: normalized image
        """
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        # type: (List[int]) -> int
        """
        choice a value of k using uniform distribution
        :param k: array of min_size
        :return: choose value
        """
        index = int(torch.empty(1).uniform_(0, float(len(k))).item())
        return k[index]

    def resize(self, image, target):
        # type:(Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        resize image to specification of a range ,and bounding box information
        :param image: norm image
        :param target: objection detection label
        :return: resized result
        """
        h, w = image.shape[-2:]
        # specify image min size
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])
        # resize image
        if torchvision._is_tracing():
            pass
        else:
            image = _resize_image(image, size, float(self.max_size))

        if target is None:
            return image, target
        # resize bounding box
        bboxes = target["bboxes"]
        bboxes = resize_bboxes(bboxes, [h, w], image.shape[-2:])
        target["bboxes"] = bboxes
        return image, target

    def max_by_axis(self, shape_list):
        # type: (List[List[int]]) -> List[int]
        max_shape = shape_list[0]
        for shape in shape_list[0]:
            for idx, item in enumerate(shape):
                max_shape[idx] = max(max_shape[idx], item)
        return max_shape

    def batch_images(self, images, size_divisible = 32):
        # type: (List[Tensor], int) -> Tensor
        """
        pack images to a batch
        :param images: images data
        :param size_divisible: adjust the image height and width to integer multiples of this value
        :return:  batch image
        """
        if torchvision._is_tracing():
            pass
        # max channel, height, width
        max_shape = self.max_by_axis([list(img.shape) for img in images])

        stride = float(size_divisible)
        max_shape[1] = int(math.ceil(float(max_shape[1]) / stride) * stride)
        max_shape[2] = int(math.ceil(float(max_shape[2]) / stride) * stride)
        batch_shape = [len(images)] + max_shape

        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

