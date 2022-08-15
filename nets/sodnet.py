from torch import nn
from nets import drn
import torch.nn.functional as F

try:
    from nn.modules import batchnormsync
except ImportError:
    pass


class SODNet(nn.Module):
    def __init__(self, backbone_name='drn_d_105', num_flabel=4, pretrained_model=None, pretrained=False):
        super(SODNet, self).__init__()
        self.backbone = drn.__dict__.get(backbone_name)(pretrained=pretrained, out_middle=True, num_classes=1000)  # 创建了初始的DRN_base model
        self.conv2 = nn.Conv2d(512, 1, 3, padding=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    

    def forward(self, im, target_size):
        ### saliency network
        x, _ = self.backbone(im)  #input: im shape [320, 320]
        x = self.conv2(x)
        x = F.interpolate(x, size=im.shape[2:], mode='bilinear', align_corners=False) #resize to the resolution of input

        pre = self.sigmoid(x)
        pre = F.interpolate(pre, size=target_size, mode='bilinear', align_corners=False) #resize to the resolution of original image
        pre = (pre - pre.min()) / (pre.max() - pre.min())
        return pre