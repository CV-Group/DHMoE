import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super(GroupedConv, self).__init__()
        self.grouped_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)

    def forward(self, x):
        return self.grouped_conv(x)

class EnhancedFocusModule(nn.Module):
    def __init__(self, in_channels, dilation_rates=[1, 2, 4, 6], squeeze_ratio=2, group_kernel_size=3, group_count=4):
        super(EnhancedFocusModule, self).__init__()

        reduced_channels = in_channels // 4
        self.branch1 = GroupedConv(in_channels, reduced_channels, kernel_size=3, dilation=dilation_rates[0], padding=dilation_rates[0], groups=group_count)
        self.branch2 = GroupedConv(in_channels, reduced_channels, kernel_size=3, dilation=dilation_rates[1], padding=dilation_rates[1], groups=group_count)
        self.branch3 = GroupedConv(in_channels, reduced_channels, kernel_size=3, dilation=dilation_rates[2], padding=dilation_rates[2], groups=group_count)
        self.branch4 = GroupedConv(in_channels, reduced_channels, kernel_size=3, dilation=dilation_rates[3], padding=dilation_rates[3], groups=group_count)

        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.Sigmoid()

        self.squeeze = nn.Conv2d(in_channels, in_channels // squeeze_ratio, kernel_size=1, bias=False)
        self.local_attention = nn.Conv2d(in_channels // squeeze_ratio, in_channels, kernel_size=group_kernel_size, stride=1, padding=group_kernel_size // 2, groups=group_count)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        concatenated = torch.cat((b1, b2, b3, b4), dim=1)

        global_features = self.global_pool(concatenated) 
        attention_weights = self.activation(global_features)  

        squeezed = self.squeeze(concatenated) 
        local_features = self.local_attention(squeezed) 

        combined_features =x + local_features

        output = self.final_conv(combined_features) * attention_weights  

        return output
