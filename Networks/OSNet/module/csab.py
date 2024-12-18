import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        mid_channels = in_channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.instance_norm = nn.InstanceNorm2d(1)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.prelu(x)
        return self.sigmoid(x)

class CSAB(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CSAB, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Channel Attention
        ca_out = self.channel_attention(x)
        ca_out = x * ca_out
        
        # Spatial Attention
        sa_out = self.spatial_attention(x)
        sa_out = x * sa_out
        
        x = ca_out + sa_out  
        return x


