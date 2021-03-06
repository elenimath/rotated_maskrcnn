# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
from skimage import util
from skimage import transform as T
import numpy as np

import numbers

import warnings
warnings.filterwarnings('ignore', message='Possible precision loss when converting from float64 to float32')


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        im_scale = 1.0
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w), im_scale

        if w < h:
            ow = size
            oh = int(size * h / w)
            im_scale = float(ow) / w
        else:
            oh = size
            ow = int(size * w / h)
            im_scale = float(oh) / h

        return (oh, ow), im_scale

    def __call__(self, image, target=None):
        size, im_scale = self.get_size(image.shape[:2])
        image =  util.img_as_float32(T.resize(image, size))
        if target is None:
            return image
        target = target.resize(image.shape[:2])
        # target.im_scale = im_scale
        target.add_field("scale", im_scale)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image[:,::-1,...]
            target = target.transpose(0)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image[::-1,...]
            target = target.transpose(1)
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = torch.from_numpy(image.transpose([2,0,1]).copy())
        return image, target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0, 3, 4]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomRotation(object):
    def __init__(self, degrees, prob=0.0):
        if isinstance(degrees, numbers.Number):
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.prob = prob

    @staticmethod
    def get_angle_uniform(degrees_range):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(min(degrees_range), max(degrees_range))

        return angle

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        angle = self.get_angle_uniform(self.degrees)
        try:  # TODO: CLEANUP
            target = target.rotate(angle)
        except ValueError:
            print("Value error thrown, skipping rotate")
            return image, target

        image = T.rotate(image, angle, center=None)
        return image, target
