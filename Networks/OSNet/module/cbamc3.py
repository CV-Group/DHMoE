import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNActiv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBNActiv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ = nn.SiLU()  # Swish activation

    def forward(self, x):
        return self.activ(self.bn(self.conv(x)))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Bottleneck(nn.Module):
    def __init__(self, in_planes,  stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        planes = in_planes//4
        self.conv1 = ConvBNActiv(in_planes, planes, kernel_size=1)
        self.conv2 = ConvBNActiv(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv3 = ConvBNActiv(planes, planes * 4, kernel_size=1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class AttentionModule(nn.Module):
    def __init__(self, in_planes,  num_blocks, stride=1):
        super(AttentionModule, self).__init__()
        downsample = None
        planes = in_planes//4
        if stride != 1 or in_planes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(Bottleneck(in_planes))
        for i in range(1, num_blocks):
            layers.append(Bottleneck(planes * 4, planes))

        self.bottleneck_layers = nn.Sequential(*layers)
        self.channel_attention = ChannelAttention(planes * 4)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.bottleneck_layers(x)
        out = self.channel_attention(out) * out
        out = self.spatial_attention(out) * out
        return out

class cbamc3(nn.Module):
    def __init__(self, in_planes, num_blocks, stride=1):
        super(cbamc3, self).__init__()
        planes = in_planes // 4
        self.conv1 = ConvBNActiv(in_planes, planes, kernel_size=3, padding=1)
        self.attention_module = AttentionModule(planes,num_blocks)
        self.conv2 = ConvBNActiv(in_planes, planes, kernel_size=1)
        self.conv3 = nn.Conv2d(planes, in_planes, kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out0 = self.attention_module(out)
        out1 = self.conv2(x)
        out = out0 + out1
        out = self.conv3(out)
        return out

# 示例使用
# if __name__ == "__main__":
#     cbamc3_layer = cbamc3(512, 1)
#     input_tensor = torch.randn(1, 512, 32, 32)
#     output = cbamc3_layer(input_tensor)
#     print("Output shape:", output.shape)
