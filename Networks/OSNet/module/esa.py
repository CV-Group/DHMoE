import torch
import torch.nn as nn
import torch.nn.functional as F

class ESA(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ESA, self).__init__()
        self.conv1x1_1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.strided_conv = nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3, stride=2, padding=1, groups=channels // reduction)
        self.pool = nn.MaxPool2d(kernel_size=7, stride=3, padding=1)
        self.depthwise_conv3x3 = nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3, padding=1, groups=channels // reduction)
        self.pointwise_conv1x1 = nn.Conv2d(channels // reduction, channels // reduction, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        # 1x1 Convolution
        x1 = self.conv1x1_1(x)
        
        # Strided Convolution and Pooling
        x2 = self.strided_conv(x1)
        x2 = self.pool(x2)
        
        # Depthwise Separable Convolution
        x2 = self.depthwise_conv3x3(x2)
        x2 = self.pointwise_conv1x1(x2)
        
        # Upsampling and 1x1 Convolution
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=False)
        x2 = self.conv1x1_2(x2)
        
        # Sigmoid Activation
        x2 = self.sigmoid(x2)
        
        # Element-wise multiplication with residual connection
        out = identity * x2
        
        return out

# 示例使用
# channels = 64
# model = ESA(channels)
# input_tensor = torch.randn(1, channels, 64, 64)  # 输入特征图
# output = model(input_tensor)
# print(output.shape)
