import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

# Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
         # Extra conv layer

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

# DAS Attention Gate
class DASAttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(DASAttentionGate, self).__init__()
        self.dsc = DepthwiseSeparableConv(in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.offset_conv = nn.Conv2d(out_channels, 18, kernel_size=3, padding=1)  # For 3x3 kernel, 18 = 2 * 3 * 3
        self.dc = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        dsc_out = self.dsc(x)
        dsc_out = self.instance_norm(dsc_out)
        dsc_out = self.dropout(dsc_out)
        offset = self.offset_conv(dsc_out)  # Compute offset
        dc_out = self.dc(dsc_out, offset)  # Pass offset to DeformConv2d
        dc_out = self.layer_norm(dc_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # Adjusted for dynamic input size
        attention_weights = self.sigmoid(dc_out)
        return dsc_out * attention_weights
    
class attention(nn.Module): 
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.conv = DepthwiseSeparableConv(in_channels, out_channels),
        self.sigmoid = nn.Sigmoid()
    def forward(self,x) :
         x0 = self.maxpool(x)
         x1 = self.avgpool(x)
         x2 = self.conv(x0)
         x3 = self.conv(x1)
         x1 = self.sigmoid(x2+x3)
         return x1

  
  
class da(nn.Module): 
    def __init__(self,in_channel, out_channel):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1, bias=False)
        self.das = DASAttentionGate(in_channel, out_channel)
        self.atten = attention(in_channel, out_channel)
        
    def forward(self,x) :
        x1, x2 = torch.split(x, 256, dim=1)
        x1 = self.das(x1)
        x2 = self.atten(x2)
        x1 = x * x1
        x2 = x * x2
        x = x1 + x2
        return x
        