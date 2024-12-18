import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
import torch.nn.functional as F
# Depthwise Separable Convolution
# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduce_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduce_ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduce_ratio, in_channels, 1, bias=False),
#             nn.Sigmoid()
#         )

    # def forward(self, x):
    #     avg_out = self.fc(self.avg_pool(x))
    #     max_out = self.fc(self.max_pool(x))
    #     return  (avg_out + max_out) / 2

# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.batch_norm1 = nn.BatchNorm2d(in_channels)
#         self.batch_norm2 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.batch_norm1(x)
#         x = self.relu(x)
#         x = self.pointwise(x)
#         x = self.batch_norm2(x)
#         x = self.relu(x)
#         return x
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        layers = []
        current_in_channels = in_channels

        for _ in range(num_layers):
            layers.append(nn.Conv2d(current_in_channels, current_in_channels, kernel_size=kernel_size, padding=padding, groups=current_in_channels))
            layers.append(nn.BatchNorm2d(current_in_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            current_in_channels = out_channels

        self.depthwise_separable_conv = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.depthwise_separable_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.depthwise_separable_conv(x)
        return x

# DAS Attention Gate
class DASAttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(DASAttentionGate, self).__init__()
        self.dsc = DepthwiseSeparableConv(in_channels, out_channels)  
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.offset_conv = nn.Conv2d(out_channels, 18, kernel_size=3, padding=1)
        self.dc = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.ca = ChannelAttention(in_channels)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        dsc_out = self.dsc(x)
        dsc_out = self.instance_norm(dsc_out)
        dsc_out = self.dropout(dsc_out)
        offset = self.offset_conv(dsc_out)
        dc_out = self.dc(dsc_out, offset)
        dc_out = self.layer_norm(dc_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attention_weights = self.sigmoid(dc_out)
        # x0 = self.ca(x)
        # x1 = x0 * x
        dc_out = attention_weights * x
        # x = x1 + dc_out
        return dc_out

class da(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.das = DASAttentionGate(in_channel , out_channel )
    def forward(self, x):
        x1 = self.das(x)
        x = x + x1 
        return x
