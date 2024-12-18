import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class FocusingTransitionModule(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(FocusingTransitionModule, self).__init__()
        
        # Global Average Pooling + MLP + Sigmoid
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, inter_channels),
            nn.ReLU(),
            nn.Linear(inter_channels, in_channels),
            nn.Sigmoid()
        )
        
        # Depthwise Separable Convolutions
        self.dyconv1 = DepthwiseSeparableConv(in_channels, in_channels)
        self.dyconv2 = DepthwiseSeparableConv(in_channels, in_channels)
        self.dyconv3 = DepthwiseSeparableConv(in_channels, in_channels)
        self.dyconv4 = DepthwiseSeparableConv(in_channels, in_channels)
        
        # Deformable Convolution + Sigmoid
        self.deform_conv = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)  # 3*3*2=18
        self.sigmoid = nn.Sigmoid()
        
        # 1x1 Convolution + Sigmoid
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid_final = nn.Sigmoid()

    def forward(self, x):
        # GAP + MLP + Sigmoid (Channel Feature Weight Optimization)
        b, c, _, _ = x.size()
        gap = self.gap(x).view(b, c)
        mlp = self.mlp(gap).view(b, c, 1, 1)
        x = x * mlp
        
        # Depthwise Separable Convolutions with Residual Connections (Dynamic Convolution)
        out1 = self.dyconv1(x) + x
        out2 = self.dyconv2(out1) + out1
        out3 = self.dyconv3(out2) + out2
        out4 = self.dyconv4(out3) + out3
        
        # Deformable Convolution and Sigmoid instead of mean operation
        offset = self.offset_conv(out4)  # 生成偏移量
        deform_out = self.deform_conv(out4, offset)
        deform_out = self.sigmoid(deform_out)
        
        # Channel-wise multiplication
        out = deform_out * mlp
        
        # 1x1 Convolution + Sigmoid
        out = self.conv1x1(out)
        out = self.sigmoid_final(out)
        
        return out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 测试网络
if __name__ == "__main__":
    ftm = FocusingTransitionModule(in_channels=512, inter_channels=128)
    x = torch.randn(1, 512, 32, 32)  # 假设输入特征图大小为 (1, 512, 32, 32)
    output = ftm(x)
    print(output.shape)  # 输出特征图大小应与输入一致 (1, 512, 32, 32)
