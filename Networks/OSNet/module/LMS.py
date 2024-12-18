import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from thop import profile


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=inp, bias=False)
        self.pointwise = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class LMultiscale(nn.Module): 
    def __init__(self,inp,dilation=[1,2,3,5],squeeze_radio=2,group_kernel_size=3,group_size=2):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(inp, inp // 4, kernel_size=3, dilation=dilation[0], padding=dilation[0])
        self.conv2 = DepthwiseSeparableConv(inp, inp // 4, kernel_size=3, dilation=dilation[1], padding=dilation[1])
        self.conv3 = DepthwiseSeparableConv(inp, inp // 4, kernel_size=3, dilation=dilation[2], padding=dilation[2])
        self.conv4 = DepthwiseSeparableConv(inp, inp // 4, kernel_size=3, dilation=dilation[3], padding=dilation[3])
        
        self.conv5 = DepthwiseSeparableConv(inp // 4, inp // 4, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigomid = nn.Sigmoid()
        self.conv = DepthwiseSeparableConv(inp,inp, kernel_size=3, padding=1)

        self.squeeze = nn.Conv2d(inp, inp // squeeze_radio, kernel_size=1, bias=False)
        self.GWC = nn.Conv2d(inp // 2, inp, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC = nn.Conv2d(inp // squeeze_radio, inp, kernel_size=1, bias=False)

    def forward(self, x):
    #多尺度提取部分
        x1= self.conv1(x)   #torch.Size([1, 128, 16, 16])
        x21= self.conv2(x)  
        x12 = x1 + x21     
        x2 = self.conv5(x12)  #torch.Size([1, 128, 16, 16])
        x32= self.conv3(x)   
        x23= x32 + x2      
        x3 = self.conv5(x23)  #torch.Size([1, 128, 16, 16])
        x43= self.conv4(x)
        x34 = x3 + x43
        x4 = self.conv5(x34)  #torch.Size([1, 128, 16, 16])
        x5 = torch.cat((x1, x2, x3, x4), dim=1)   #torch.Size([1, 512, 16, 16])
   # # 权重注意
        x6_0 = self.pool(x5)   #torch.Size([1, 512, 1, 1])
        x6 = self.conv(x6_0)   #torch.Size([1, 512, 1, 1])
        x7 = self.sigomid(x6)  #torch.Size([1, 512, 1, 1])
    
    # 局部细节感知
        x8 =  self.squeeze(x5)
        x9 = self.GWC(x8)
        x10 = self.PWC(x8) 
        x11 = x9 + x10 
        x = x11 * x7
        return x
    


# net  = LMultiscale(512).cuda()
# input = torch.randn(1, 512, 16, 16).cuda()
# flops, params = profile(net, inputs=(input,))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')
       







