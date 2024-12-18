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

class ChannelAggregation(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAggregation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.gelu = nn.GELU()
        self.scale_factor = nn.Parameter(torch.ones(1))  # 可训练的缩放因子
    
    def forward(self, y):
        conv_y = self.conv1(y)
        gelu_y = self.gelu(conv_y)
        subtracted = y - gelu_y
        scaled = self.scale_factor * subtracted
        aggregated = y + scaled
        return aggregated

class CAB(nn.Module):
    def __init__(self, in_channels, reduction=2):  # 将reduction改为2
        super(CAB, self).__init__()
        expanded_channels = in_channels * reduction
        self.conv1 = nn.Conv2d(in_channels, expanded_channels, kernel_size=1)
        self.dwconv3x3 = DepthwiseSeparableConv(expanded_channels, expanded_channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.channel_aggregation = ChannelAggregation(expanded_channels)
        self.conv2 = nn.Conv2d(expanded_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        # Y = GELU(DWConv3x3(Conv1x1(X)))
        y = self.conv1(x)
        y = self.dwconv3x3(y)
        y = self.gelu(y)
        
        # H_CA(Y)
        h_ca = self.channel_aggregation(y)
        
        # F_CAB = Conv1x1(H_CA(Y)) + X
        f_cab = self.conv2(h_ca) + x
        
        return f_cab

# # 测试网络
# if __name__ == "__main__":
#     cab = CAB(in_channels=64)  # 确保输入通道数为64
#     x = torch.randn(1, 64, 32, 32)  # 假设输入特征图大小为 (1, 64, 32, 32)
#     output = cab(x)
#     print(output.shape)  # 输出特征图大小
