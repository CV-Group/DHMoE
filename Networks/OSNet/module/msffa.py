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

class FusionFeatureAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(FusionFeatureAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MsFFA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MsFFA, self).__init__()
        reduced_channels = in_channels // 4
        out_channels = in_channels // 4
        self.split = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # Split
        
        self.conv1x1 = nn.Conv2d(reduced_channels, out_channels, kernel_size=1)
        self.conv3x3 = DepthwiseSeparableConv(reduced_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = DepthwiseSeparableConv(reduced_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = DepthwiseSeparableConv(reduced_channels, out_channels, kernel_size=7, padding=3)
        
        self.fusion_attention = FusionFeatureAttention(out_channels, reduction)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        split_channels = channels // 4
        
        # Split the input into four parts
        x_split = self.split(x)
        x0, x1, x2, x3 = torch.split(x_split, split_channels, dim=1)
        
        # Apply convolutions
        f0 = self.conv1x1(x0)
        f1 = self.conv3x3(x1)
        f2 = self.conv5x5(x2)
        f3 = self.conv7x7(x3)
        
        # Fusion Feature Attention
        f0_att = self.fusion_attention(f0)
        f1_att = self.fusion_attention(f1)
        f2_att = self.fusion_attention(f2)
        f3_att = self.fusion_attention(f3)
        
        # Re-weight and concatenate
        o0 = f0 * f0_att
        o1 = f1 * f1_att
        o2 = f2 * f2_att
        o3 = f3 * f3_att
        
        out = torch.cat([o0, o1, o2, o3], dim=1)
        return out

