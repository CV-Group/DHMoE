import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, in_channels // reduction)
        
        self.conv1 = DepthwiseSeparableConv(in_channels, mip, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()
        
        self.conv_h = DepthwiseSeparableConv(mip, in_channels, kernel_size=3, padding=1)
        self.conv_w = DepthwiseSeparableConv(mip, in_channels, kernel_size=3, padding=1)

        self.offset_conv = nn.Conv2d(mip, 2 * 3 * 3, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(mip, mip, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        offset = self.offset_conv(y)
        y = self.deform_conv(y, offset)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_h * a_w
        return out


