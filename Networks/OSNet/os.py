from __future__ import division, absolute_import
import torch
from torch import nn
from torch.nn import functional as F
from thop import profile
from torchsummary import summary
# from .module.LMS import LMultiscale
# from .module.LMS_256 import LMultiscale 
# from .module.atten2 import ghost
# from .RecursiveFocusingModule import CombinedConvAttention as RFM
from .RecursiveFocusingModule import EnhancedMultiScaleModule as RFM

class ConvLayer(nn.Module):
    """Convolution layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution without non-linearity."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='sigmoid',
        reduction=16,
        layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = x3 + residual
        return F.relu(out)


##########
# Network architecture
##########
class BaseNet(nn.Module):

    def _make_layer(
        self, block, layer, in_channels, out_channels, reduce_spatial_size
    ):
        layers = []

        layers.append(block(in_channels, out_channels))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class OSNet(BaseNet):

    def __init__(
        self,
        num_classes,
        blocks,
        layers,
        channels,
        feature_dim=512,
        loss='softmax',
        pool='avg',
        **kwargs
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss

        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False
        )
        self.conv5 = Conv1x1(channels[3], channels[3])

    def forward(self, x):
        # t_outputs = []
        x = self.conv1(x)
        # t_outputs.append(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        # t_outputs.append(x)
        x = self.conv3(x)
        # t_outputs.append(x)
        x = self.conv4(x)
        # t_outputs.append(x)
        x = self.conv5(x)
        # t_outputs.append(x)
        
        return x
    # def forward(self, x):   #torch.Size([2, 3, 256, 256])
    #     x = self.conv1(x)   #torch.Size([2, 64, 128, 128])
    #     x = self.maxpool(x) #torch.Size([2, 64, 64, 64])
    #     x = self.conv2(x)   #torch.Size([2, 256, 32, 32])
    #     x1 = self.conv3(x)   #torch.Size([2, 384, 16, 16])
    #     x2 = self.conv4(x1)   #torch.Size([2, 512, 16, 16])
    #     x3 = self.conv5(x2)   #torch.Size([2, 512, 16, 16])
    #     return x1,x2,x3
    


def load_osnet_weights():
  # Create an instance of the MobileViT model
  net = OSNet(
        num_classes=1000,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        # channels=[48, 192, 288, 384],
        # channels=[32, 128, 192, 256],
        loss='softmax',
        # conv1_IN=True,
        # **kwargs
    )
  model_path = '/home/jingan/wangluo/C3_LWN/models/SCC_Model/OSNet/osnet_x1_0_imagenet.pth'
#   model_path = '/home/jingan/wangluo/counting-total/Networks/OSNet/osnet_x0_75_imagenet.pth'
#   model_path = '/home/jingan/wangluo/counting-total/Networks/OSNet/osnet_x0_5_imagenet.pth'

  state_dict = torch.load(model_path)
#   print(state_dict.keys())
  state_dict.pop("fc.0.weight")
  state_dict.pop("fc.0.bias")
  state_dict.pop("fc.1.weight")
  state_dict.pop("fc.1.bias")
  state_dict.pop("fc.1.running_mean")
  state_dict.pop("fc.1.running_var")
  state_dict.pop("fc.1.num_batches_tracked")
  state_dict.pop("classifier.weight")
  state_dict.pop("classifier.bias")

  net.load_state_dict(state_dict)
  
  
  return net


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.vgg = load_osnet_weights()

        # num_filters = [16, 32, 64, 64]
        # num_filters = [32, 64, 64]
        # # self.RFM1 = RFM(256, 128,num_filters[0])
        # self.RFM2 = RFM(384, 192, num_filters[0])
        # self.RFM3 = RFM(512, 256, num_filters[1])
        # self.RFM4 = RFM(512, 256, num_filters[2])
        
        
        # self.fusion = Fusion(num_filters[0],num_filters[1],num_filters[2])

        self.de_pred = nn.Sequential(
                                    nn.ConvTranspose2d(512,256,4,stride=2,padding=1,output_padding=0,bias=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(256,128,4,stride=2,padding=1,output_padding=0,bias=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(128,64,4,stride=2,padding=1,output_padding=0,bias=True),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(64,1,4,stride=2,padding=1,output_padding=0,bias=True),
                                    nn.ReLU(),
                                    )
        # self.de_pred = nn.Sequential(
        #                     nn.ConvTranspose2d(384,192,4,stride=2,padding=1,output_padding=0,bias=True),
        #                     nn.ReLU(),
        #                     nn.ConvTranspose2d(192,96,4,stride=2,padding=1,output_padding=0,bias=True),
        #                     nn.ReLU(),
        #                     nn.ConvTranspose2d(96,48,4,stride=2,padding=1,output_padding=0,bias=True),
        #                     nn.ReLU(),
        #                     nn.ConvTranspose2d(48,1,4,stride=2,padding=1,output_padding=0,bias=True),
        #                     )
        # self.de_pred = nn.Sequential(
        #                     nn.ConvTranspose2d(256,128,4,stride=2,padding=1,output_padding=0,bias=True),
        #                     nn.ReLU(),
        #                     nn.ConvTranspose2d(128,64,4,stride=2,padding=1,output_padding=0,bias=True),
        #                     nn.ReLU(),
        #                     nn.ConvTranspose2d(64,32,4,stride=2,padding=1,output_padding=0,bias=True),
        #                     nn.ReLU(),
        #                     nn.ConvTranspose2d(32,1,4,stride=2,padding=1,output_padding=0,bias=True),
        #                     )
        # self.de_pred = nn.Sequential(
        #                     nn.ConvTranspose2d(160,80,4,stride=2,padding=1,output_padding=0,bias=True),
        #                     nn.ReLU(),
        #                     nn.ConvTranspose2d(80,40,4,stride=2,padding=1,output_padding=0,bias=True),
        #                     nn.ReLU(),
        #                     nn.ConvTranspose2d(40,20,4,stride=2,padding=1,output_padding=0,bias=True),
        #                     nn.ReLU(),
        #                     nn.ConvTranspose2d(20,1,4,stride=2,padding=1,output_padding=0,bias=True),
        #                     nn.ReLU(),
        #                     )
        

    def forward(self, x):  
        t_outputs = self.vgg(x)
        # x1 = self.RFM1(x1)
        # x1 = self.RFM2(t_outputs[2])
        # x2 = self.RFM3(t_outputs[3])
        # x3 = self.RFM4(t_outputs[4])
        
        # x = torch.cat([x1, x2, x3], dim=1)         
     
        x = self.de_pred(t_outputs)

        return x


# net = decoder().cuda()
# input = torch.randn(1, 3, 256, 256).cuda()
# flops, params = profile(net, inputs=(input,))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')