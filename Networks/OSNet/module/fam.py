import torch
import torch.nn as nn
import torch.nn.functional as F

class FCA(nn.Module):
    def __init__(self, channels, reduction=16):
        super(FCA, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        std_out = torch.std(x, dim=[2, 3], keepdim=True)
        
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        std_out = self.fc(std_out)
        
        x0 = x * avg_out
        x1 = x * max_out
        x2 = x * std_out
        x = x0 + x1 + x2
        
        return x
class FSA(nn.Module):
    def __init__(self, channels):
        super(FSA, self).__init__()
        self.channels = channels
        # 使用分组卷积减少参数量
        self.conv7x7 = nn.Conv2d(channels * 3, channels, kernel_size=7, padding=3, groups=channels // 8, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        std_out = torch.std(x, dim=[2, 3], keepdim=True)

        # 在通道维度拼接
        out = torch.cat([avg_out, max_out, std_out], dim=1)
        
        out = self.conv7x7(out)
        out = self.sigmoid(out)

        return x * out

class FAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(FAM, self).__init__()
        self.fca = FCA(channels, reduction)
        self.fsa = FSA(channels)
    
    def forward(self, x):
        fca_out = self.fca(x)
        x = x + fca_out
        fsa_out = self.fsa(x)
        x = x + fsa_out
        return x

# # 示例使用
# channels = 512
# model = FAM(channels)
# input_tensor = torch.randn(1, channels, 32, 32)  # 输入特征图
# output = model(input_tensor)
# print(output.shape)
