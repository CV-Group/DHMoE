import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ghost(nn.Module):
    def __init__(self, inp, oup):
        super(ghost, self).__init__()
        mid_channels = inp // 2
        self.conv = nn.Conv2d(inp, mid_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dwc = DepthwiseSeparableConv(mid_channels, mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.offset_conv = nn.Conv2d(mid_channels * 2, 18, kernel_size=3, padding=1)
        self.dc = DeformConv2d(mid_channels * 2, oup, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.conv(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.dwc(x0)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        x2 = torch.cat([x0, x1], dim=1)
        offset = self.offset_conv(x2)
        x2 = self.dc(x2, offset)
        weights = self.sigmoid(x2)
        return x * weights
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.ops import DeformConv2d

# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.bn = nn.GroupNorm(1, out_channels)  # Using GroupNorm instead of BatchNorm2d

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return x

# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
#         self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

#     def forward(self, x):
#         scale = F.adaptive_avg_pool2d(x, (1, 1))
#         scale = F.relu(self.fc1(scale))
#         scale = torch.sigmoid(self.fc2(scale))
#         return x * scale

# class ghost(nn.Module):
#     def __init__(self, inp, oup):
#         super(ghost, self).__init__()
#         mid_channels = inp // 2
#         self.conv = nn.Conv2d(inp, mid_channels, kernel_size=1, padding=0)
#         self.gn1 = nn.GroupNorm(1, mid_channels)
#         self.swish = nn.SiLU()  # Swish activation function
#         self.dwc = DepthwiseSeparableConv(mid_channels, mid_channels)
#         self.offset_conv = nn.Conv2d(mid_channels * 2, 18, kernel_size=3, padding=1)  # Reduced output channels
#         self.dc = DeformConv2d(mid_channels * 2, oup, kernel_size=3, padding=1)
#         self.se = SEBlock(oup)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x0 = self.conv(x)
#         x0 = self.gn1(x0)
#         x0 = self.swish(x0)
#         x1 = self.dwc(x0)
#         x2 = torch.cat([x0, x1], dim=1)
#         offset = self.offset_conv(x2)
#         x2 = self.dc(x2, offset)
#         x2 = self.se(x2)
#         weights = self.sigmoid(x2)
#         return x * weights


