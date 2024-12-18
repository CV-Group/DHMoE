import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class FeatureDecomposition(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(FeatureDecomposition, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.scale_factor = nn.Parameter(torch.zeros(1))  # 可训练的缩放因子
        self.gelu = nn.GELU()
    
    def forward(self, x):
        y = self.conv1(x)
        gap_y = self.gap(y)
        subtracted = y - gap_y
        scaled = self.scale_factor * subtracted
        z = y + scaled
        z = self.gelu(z)
        return z

class MultiOrderGatedAggregation(nn.Module):
    def __init__(self, in_channels):
        super(MultiOrderGatedAggregation, self).__init__()
        self.dwconv1 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=5, padding=2, dilation=1)
        self.dwconv2 = DepthwiseSeparableConv(in_channels * 3 // 8, in_channels * 3 // 8, kernel_size=5, padding=4, dilation=2)  # 扩张率为2
        self.dwconv3 = DepthwiseSeparableConv(in_channels // 2, in_channels // 2, kernel_size=7, padding=9, dilation=3)  # 扩张率为3
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gelu2 = nn.GELU()
    
    def forward(self, z):
        y_c = self.dwconv1(z)
        channels = y_c.shape[1]
        y_c1, y_c2, y_c3 = torch.split(y_c, [channels // 8, 3 * channels // 8, channels // 2], dim=1)
        
        y_c2 = self.dwconv2(y_c2)
        y_c3 = self.dwconv3(y_c3)
        
        y_c_concat = torch.cat((y_c1, y_c2, y_c3), dim=1)
        
        y_c1 = self.conv1(y_c_concat)
        y_c1 = self.gelu1(y_c1)
        
        y_c2 = self.conv2(z)
        y_c2 = self.gelu2(y_c2)
        
        x = y_c2 * y_c1
        x = self.conv3(x)
        
        return x


class SAB(nn.Module):
    def __init__(self, in_channels):
        super(SAB, self).__init__()
        self.fd = FeatureDecomposition(in_channels)
        self.moga = MultiOrderGatedAggregation(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        h_fd = self.fd(x)
        h_moga = self.moga(h_fd)
        f_sab = self.conv1(h_moga) + x
        return f_sab

# # 测试网络
# if __name__ == "__main__":
#     sab = SAB(in_channels=64)  # 确保输入通道数为64
#     x = torch.randn(1, 64, 32, 32)  # 假设输入特征图大小为 (1, 64, 32, 32)
#     output = sab(x)
#     print(output.shape)  # 输出特征图大小
