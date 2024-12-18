import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, padding=1, dilation=1, reduction=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(inp, inp // reduction, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=inp // reduction, bias=False)
        self.pointwise = nn.Conv2d(inp // reduction, oup, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class RCAB(nn.Module):
    def __init__(self, in_channels, reduction=16, conv_reduction=4):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // conv_reduction, kernel_size=3, padding=1, bias=False)
        self.gelu = nn.GELU()
        self.depthwise_conv = DepthwiseSeparableConv(in_channels // conv_reduction, in_channels // conv_reduction, kernel_size=3, padding=1, reduction=conv_reduction)
        self.pointwise_conv = nn.Conv2d(in_channels // conv_reduction, in_channels, kernel_size=1, bias=False)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # Residual path
        residual = x
        
        # Convolution + GELU + Depthwise Separable Convolution
        out = self.conv1(x)
        out = self.gelu(out)
        out = self.depthwise_conv(out)
        out = self.pointwise_conv(out)
        
        # Global pooling + Fully Connected layers + Sigmoid
        attention = self.global_pool(out)
        attention = self.fc1(attention)
        attention = self.gelu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        
        # Element-wise multiplication
        out = out * attention
        
        # Batch normalization
        out = self.batch_norm(out)
        
        # Element-wise sum
        out = out + residual
        
        return out

# 示例使用
# channels = 64
# model = RCAB(channels)
# input_tensor = torch.randn(1, channels, 32, 32)  # 输入特征图
# output = model(input_tensor)
# print(output.shape)
