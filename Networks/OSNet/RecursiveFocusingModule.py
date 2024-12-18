import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):  
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=self.avg_pool(x)
        avgout = self.shared_MLP(x)
        return self.sigmoid(avgout)
    
class SpatialAttention(nn.Module):  
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.conv1(x)
        return self.sigmoid(x)    



class CombinedConvAttention(nn.Module):
    def __init__(self, channel, channel2, num_filters):
        super(CombinedConvAttention, self).__init__()
        self.ch_att_s = ChannelAttention(channel) 
        self.depthwise_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel)
        self.pointwise_conv = nn.Conv2d(channel, channel2, kernel_size=1)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(channel+channel2, channel2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channel2))
        self.output_conv = nn.Conv2d(channel2, num_filters, kernel_size=3, stride=1, padding=1)
        self.norm_activation = nn.Sequential(
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU())

    def forward(self, x):
        x = self.ch_att_s(x) * x   #torch.Size([2, 256, 32, 32])
        dw = self.depthwise_conv(x)  #torch.Size([2, 256, 32, 32])
        pw = self.pointwise_conv(dw)    #torch.Size([2, 128, 32, 32])
        concatenated = torch.cat([x, pw], dim=1)  #torch.Size([2, 384, 32, 32])
        fused = self.feature_fusion(concatenated)  #torch.Size([2, 128, 32, 32])
        out = self.output_conv(fused)   #torch.Size([2, 16, 32, 32])
        result = self.norm_activation(out) #torch.Size([2, 16, 32, 32])
        return result

# class Fusion(nn.Module):
#     def __init__(self, num_filters1, num_filters2, num_filters3, num_filters4):
#         super(Fusion, self).__init__()
#         self.upsample_1 = nn.ConvTranspose2d(in_channels=num_filters2, out_channels=num_filters2, kernel_size=4, padding=1, stride=2)
#         self.upsample_2 = nn.ConvTranspose2d(in_channels=num_filters3, out_channels=num_filters3, kernel_size=4, padding=1, stride=2)
#         self.upsample_3 = nn.ConvTranspose2d(in_channels=num_filters4, out_channels=num_filters4, kernel_size=4, padding=1, stride=2)
#         # self.final = nn.Sequential(
#         #     nn.Conv2d(num_filters1+num_filters2+num_filters3+num_filters4, 1, kernel_size=1, padding=0),
#         #     nn.ReLU(),
#         # )
        
#     def forward(self, x1, x2, x3, x4):
#         x2 = self.upsample_1(x2)
#         x3 = self.upsample_2(x3)
#         x4 = self.upsample_3(x4)

#         x = torch.cat([x1, x2, x3, x4], dim=1)
#         # x = self.final(x)
        
#         return x




class EnhancedMultiScaleModule(nn.Module):
    def __init__(self, channel, channel2, num_filters):
        super(EnhancedMultiScaleModule, self).__init__()
        self.spatial_att = SpatialAttention(kernel_size=3)
        
        # 使用深度可分离卷积增强模块
        self.conv1x1 = nn.Conv2d(channel, channel2, kernel_size=1)
        self.dilated_dw_conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2, dilation=2, groups=channel),
            nn.Conv2d(channel, channel2, kernel_size=1)  # Point-wise convolution
        )
        self.dilated_dw_conv5x5 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=4, dilation=2, groups=channel),
            nn.Conv2d(channel, channel2, kernel_size=1)  # Point-wise convolution
        )

        self.concat_conv = nn.Conv2d(channel2 , num_filters, kernel_size=1)
        self.final_activation = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(num_features=num_filters))

    def forward(self, x):
        x = self.spatial_att(x) * x
        conv1 = self.conv1x1(x)
        conv3 = self.dilated_dw_conv3x3(x)
        conv5 = self.dilated_dw_conv5x5(x)
        # concatenated = torch.cat([conv1, conv3, conv5], dim=1)
        concatenated = conv1 + conv3 + conv5
        fused = self.concat_conv(concatenated)
        result = self.final_activation(fused)
        return result


