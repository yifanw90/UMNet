from __future__ import division
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from collections import namedtuple


class Transform(object):
    def __init__(self, config):
        transforms = []
        transforms.append(ToTensor())
        transforms.append(Normalize(config.input_mean, config.input_std))
        transforms.append(Resize(config.input_size))
        self.transforms = T.Compose(transforms)
    def __call__(self, batch):
        batch = self.transforms(batch)
        return batch


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self):
        self.to_tensor = T.ToTensor()
        self.dtype = torch.float32

    def __call__(self, image):
        image = self.to_tensor(image).type(self.dtype)  # C, H, W
        
        if image.shape[0] == 1:
            image = torch.cat((image, image, image), dim=0)

        return image


class Normalize(object):
    def __init__(self, mean, std):
        self.normalize = T.Normalize(mean, std, inplace=True)
    def __call__(self, image):
        return self.normalize(image)


class Resize(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, image):
        return F.interpolate(image[None], size=self.input_size, mode='bilinear', align_corners=False)[0]
