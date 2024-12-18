import torch
import torch.nn as nn
import torch.nn.functional as F

class GCE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCE, self).__init__()
        self.gh = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gd = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        gh = self.gh(x)
        gv = self.gv(x)
        gd = self.gd(x)
        out = torch.cat([gh, gv, gd], dim=1)
        out = self.conv1x1(out)
        return out

class GFFRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GFFRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.gce = GCE(in_channels, out_channels)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        
        gce_out = self.gce(x)
        
        out = self.conv1x1(out)
        out += residual + gce_out
        
        return out

class DRR(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(DRR, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.gffrb_blocks = nn.Sequential(
            *[GFFRB(out_channels, out_channels) for _ in range(num_blocks)]
        )
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.relu_in(out)
        out = self.gffrb_blocks(out)
        out = self.conv_out(out)
        out = self.pixel_shuffle(out)
        return out

# Example of usage:
# model = DRR(in_channels=3, out_channels=64, num_blocks=5)
# x = torch.randn(1, 3, 32, 32)  # Example input
# output = model(x)
# print(output.shape)
