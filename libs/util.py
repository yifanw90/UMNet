import numpy as np
from PIL import Image 
import os


def load_img_names(img_path):
    img_names = os.listdir(img_path)
    img_names = [i for i in img_names if i.endswith('.png') or i.endswith('.jpg')] 
    return img_names


def save_img(im_np, im_name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    im = Image.fromarray(im_np)
    im.save(os.path.join(save_path, im_name))


class eval_dataset:
    def __init__(self, image_root, gt_root):
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith('.png')]
        self.image_root = image_root
        self.gt_root = gt_root
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        image = self.binary_loader(os.path.join(self.image_root, self.img_list[self.index] + '.png'))
        gt = self.binary_loader(os.path.join(self.gt_root, self.img_list[self.index] + '.png'))
        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')