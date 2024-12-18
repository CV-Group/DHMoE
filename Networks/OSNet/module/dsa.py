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

class PSA(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(PSA, self).__init__()
        reduced_channels = in_channels // reduction
        self.conv1 = DepthwiseSeparableConv(in_channels, reduced_channels, dilation=1, padding=1)
        self.conv2 = DepthwiseSeparableConv(in_channels, reduced_channels, dilation=3, padding=3)
        self.conv3 = DepthwiseSeparableConv(in_channels, reduced_channels, dilation=5, padding=5)
        self.conv_out = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)  # 将通道数恢复为原始输入的通道数
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        reduced_channels = channels // 8
        num_pixels = height * width

        # Fp1, Fp2, Fp3
        Fp1 = self.conv1(x).view(batch_size, reduced_channels, num_pixels)
        Fp2 = self.conv2(x).view(batch_size, reduced_channels, num_pixels)
        Fp3 = self.conv3(x).view(batch_size, reduced_channels, num_pixels)

        # 打印张量形状以进行调试
        # print("Fp1 shape:", Fp1.shape)
        # print("Fp2 shape:", Fp2.shape)
        # print("Fp3 shape:", Fp3.shape)
        
        # Transpose Fp2 for matrix multiplication
        Fp2 = Fp2.permute(0, 2, 1)
        
        # Multiplication and Softmax for Mpam
        Mpam = torch.matmul(Fp2, Fp3)
        Mpam = F.softmax(Mpam, dim=-1)
        
        # Fpmap
        Fpmap = torch.matmul(Fp1, Mpam.permute(0, 2, 1))
        Fpmap = Fpmap.view(batch_size, reduced_channels, height, width)
        
        # 恢复通道数
        Fpmap = self.conv_out(Fpmap)
        
        # Combine
        out = x + Fpmap
        
        return out


class CSA(nn.Module):
    def __init__(self, in_channels):
        super(CSA, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, in_channels, dilation=1, padding=1)
        self.conv2 = DepthwiseSeparableConv(in_channels, in_channels, dilation=3, padding=3)
        self.conv3 = DepthwiseSeparableConv(in_channels, in_channels, dilation=5, padding=5)
        self.lambda_c = nn.Parameter(torch.zeros(1))  # 可训练的注意力系数
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        num_pixels = height * width
        
        # Fc1, Fc2, Fc3
        Fc1 = self.conv1(x).view(batch_size, channels, num_pixels)
        Fc2 = self.conv2(x).view(batch_size, channels, num_pixels)
        Fc3 = self.conv3(x).view(batch_size, channels, num_pixels)
        
        # Transpose Fc2 for matrix multiplication
        Fc2 = Fc2.permute(0, 2, 1)
        
        # Multiplication and Softmax for Mcam
        Mcam = torch.matmul(Fc1, Fc2)
        Mcam = F.softmax(Mcam, dim=-1)
        
        # Fcmap
        Fcmap = torch.matmul(Mcam, Fc3)
        Fcmap = Fcmap.view(batch_size, channels, height, width)
        
        # Combine
        F_cmap_prime = self.lambda_c * Fcmap + x
        
        return F_cmap_prime


class DSA(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(DSA, self).__init__()
        self.psa = PSA(in_channels, reduction)
        self.csa = CSA(in_channels)
    
    def forward(self, x):
        # Apply PSA
        x_psa = self.psa(x)
        
        # Apply CSA
        x_csa = self.csa(x)
        x = x_csa + x_psa
        return x

# # 测试网络
# if __name__ == "__main__":
#     dsa = DSA(in_channels=256)  # 确保输入通道数为256
#     x = torch.randn(1, 256, 16, 16)  # 假设输入特征图大小为 (1, 256, 16, 16)
#     output = dsa(x)
    # print(output.shape)  # 输出特征图大小

    # print(output.shape)  # 输出特征图大小
