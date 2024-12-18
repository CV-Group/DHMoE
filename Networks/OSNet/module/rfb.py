import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MultiScaleDilatedConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleDilatedConvModule, self).__init__()
        
        # 1x1 Conv
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Different scale Convs
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv3x3 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=7, padding=3)
        
        # Dilated Convs
        self.dconv3x3 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dconv5x5_1 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=5, padding=4, dilation=2)
        self.dconv5x5_2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=5, padding=4, dilation=2)
        
        # Final 3x3 Conv
        self.final_conv3x3 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 1x1 Conv
        x1 = self.conv1x1(x)
        
        # Different scale Convs
        x1_1 = self.conv1x1_2(x1)
        x3 = self.conv3x3(x1)
        x5 = self.conv5x5(x1)
        x7 = self.conv7x7(x1)
        
        # Dilated Convs
        d3 = self.dconv3x3(x3)
        d5_1 = self.dconv5x5_1(x5)
        d5_2 = self.dconv5x5_2(x7)
        
        # Concat
        out = torch.cat([x1_1, d3, d5_1, d5_2], dim=1)
        
        # Final 3x3 Conv
        out = self.final_conv3x3(out)
        
        return out

# # 测试网络
# if __name__ == "__main__":
#     msdc = MultiScaleDilatedConvModule(in_channels=512, out_channels=256)
#     x = torch.randn(1, 512, 128, 128)  # 假设输入特征图大小为 (1, 512, 128, 128)
#     output = msdc(x)
#     print(output.shape)  # 输出特征图大小
