import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from nets.sodnet import SODNet
from libs.util import load_img_names, save_img
from libs.dataset import Transform
from config import config


def test():   
    transforms = Transform(config)
    sal_net = SODNet().cuda()
    sal_net.load_state_dict(torch.load(config.model_path))

    for dataset in config.datasets:
        print('Test on dataset: %s'%(dataset))
        img_path = os.path.join(config.data_path, dataset, 'img')  
        img_names = load_img_names(img_path)

        save_path = os.path.join(config.save_path, dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)           
               
        sal_net.eval()
        for img_id, img_name in enumerate(img_names):
            print('predicting for %d / %d' % (img_id + 1, len(img_names)))
            
            im_ori = Image.open(os.path.join(img_path, img_name))
            target_size = im_ori.size[::-1]
            im = transforms(im_ori)
            im = im[None].cuda()
          
            with torch.no_grad():
                mask_pre = sal_net(im, target_size) 
                mask_pre = mask_pre.cpu().numpy()[0][0]
                save_img(np.uint8(mask_pre*255), img_name[:-4]+'.png', save_path)
   
            
if __name__ == '__main__':
    test()
