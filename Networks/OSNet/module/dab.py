import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvGroup(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvGroup, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=1, padding=0)
        self.conv3 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv4 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.sigmoid(out)
        x = self.conv4(x)
        return out * x

class GAP_FRF_Softmax(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(GAP_FRF_Softmax, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, 2 * in_channels)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        gap = self.gap(x).view(x.size(0), -1)
        frf = F.relu(self.fc1(gap))
        alpha_beta = self.fc2(frf).view(x.size(0), 2, x.size(1))
        alpha_beta = self.softmax(alpha_beta)
        alpha, beta = alpha_beta[:, 0, :], alpha_beta[:, 1, :]
        alpha = alpha.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return alpha, beta

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=1)
        self.conv3 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.sigmoid(out)
        return out * x

class DAB(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(DAB, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_group = ConvGroup(out_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
        self.ca = GAP_FRF_Softmax(out_channels, reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, 2 * in_channels)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
            # Step 1: Initial Convolution
            x0 = self.conv1(x)
            
            # Step 2: Feature Extraction
            f2 = self.conv_group(x0)
            
            # Step 3: Channel Attention Branch
            # f1 = self.ca(f2)
            
            # Step 4: Dynamic Attention Weights
            gap = self.gap(x0).view(x0.size(0), -1)
            frf = F.relu(self.fc1(gap))
            alpha_beta = self.softmax(self.fc2(frf)).view(x.size(0), 2, -1)
            alpha, beta = alpha_beta[:, 0, :], alpha_beta[:, 1, :]
            alpha = alpha.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
            
            # Step 5: Dynamic Attention Enhanced Fusion
            f3 = alpha * x0 + beta * f2
            f_dab = self.conv2(f3) + x0
            
            return f_dab

# # 示例使用
# dab = DAB(in_channels=64, out_channels=32)  # 减少out_channels以进一步减少参数
# input_tensor = torch.randn(1, 64, 32, 32)
# output = dab(input_tensor)
# print("Output shape:", output.shape)
