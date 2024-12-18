import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from .common import default_conv, BasicBlock

def same_padding(images, ksize, stride): 
    '''
    For Convolution.
    '''
    _, _, rows, cols = images.size() 
    padding_rows = max(0, ((rows + stride - 1) // stride -1)*stride+ksize-rows)
    padding_cols = max(0, ((cols + stride - 1) // stride -1)*stride+ksize-cols)
    # Pad the inp
    pd_top = int(padding_rows / 2.)
    pd_left = int(padding_cols / 2.)
    pd_bottom = padding_rows - pd_top
    pd_right = padding_cols - pd_left  
    return torch.nn.ZeroPad2d((pd_left, pd_right, pd_top, pd_bottom))(images)

def extract_image_patches(images, ksize, stride, padding='same'):  
    if padding == 'same':
        images = same_padding(images, ksize, stride)     
    return nn.Unfold(ksize, 1, 0, stride)(images)

def get_random_patches(imgs, p_size, patch_n):
    n, c, h, w = imgs.shape
    i_h = np.random.choice(range(0, h - p_size), size=[patch_n, 1], replace=False)
    i_w = np.random.choice(range(0, w - p_size), size=[patch_n, 1], replace=False)

    i_h = np.linspace(i_h, i_h + p_size - 1, p_size, dtype=int, axis=1)
    i_h = np.repeat(i_h, p_size, 2).flatten()
    i_w = np.repeat(i_w, p_size, 1)
    i_w = np.linspace(i_w, i_w + p_size - 1, p_size, dtype=int, axis=2).flatten()
    
    return imgs[:, :, i_h, i_w].reshape(n, c, patch_n, p_size, p_size) 

# Channel attention 
class CA(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()  

        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), nn.Sigmoid()
        )
    def forward(self, x):   
        return x * self.conv_du(self.avg_pool(x))

 
class SRNL(nn.Module):
    def __init__(self, channel=512, reduction=2, conv=default_conv):
        super().__init__()
        self.channel = channel
        self.redu_c = channel // reduction
        self.conv_match1 = BasicBlock(conv, channel, self.redu_c, 1, act=nn.PReLU())
        self.conv_match2 = BasicBlock(conv, channel, self.redu_c, 1, act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, act=nn.PReLU())
        self.softmax = nn.Softmax(1)
        self.patch_size = 48 #############
        
    def __train_forward(self, x):
        N, _, H, W = x.shape 
        x_embed_1 = self.conv_match1(x).reshape(N, -1, H * W) 
        x_embed_2 = self.conv_match2(x).reshape(N, -1, H * W).permute(0, 2, 1) 
        x_assembly = self.conv_assembly(x).reshape(N, -1, H * W) 
        score = self.softmax(torch.matmul(x_embed_2, x_embed_1))  
        return torch.matmul(x_assembly, score).reshape(N, -1, H, W)

    def __infer_forward(self, x):
        N, C, H, W = x.shape   
        
        if H <= self.patch_size or W <= self.patch_size:
            return self.__train_forward(x) 

        patches1 = F.unfold(x, self.patch_size, stride=self.patch_size)
        patches2 = F.unfold(x[:, :, -self.patch_size:, :], self.patch_size, stride=self.patch_size)
        patches3 = F.unfold(x[:, :, :, -self.patch_size:], self.patch_size, stride=self.patch_size)
        patch4 = x[:, :, -self.patch_size:, -self.patch_size:].reshape(N, -1, 1)
        patches = torch.cat([patches1, patches2, patches3, patch4], -1).permute(0, 2, 1).reshape(-1, C, self.patch_size, self.patch_size) 
        n1, n2, n3 = patches1.size(-1), patches2.size(-1), patches3.size(-1)
        n = n1 + n2 + n3 + 1

        embed1 = self.conv_match1(patches).reshape(N * n, -1, self.patch_size * self.patch_size) 
        embed2 = self.conv_match2(patches).reshape(N * n, -1, self.patch_size * self.patch_size).permute(0, 2, 1) 
        assem = self.conv_assembly(patches).reshape(N * n, -1, self.patch_size * self.patch_size) 

        score = self.softmax(torch.matmul(embed2, embed1)) 
        out_patches = torch.matmul(assem, score).reshape(N, n, -1).permute(0, 2, 1)

        out_patches1, out_patches2, out_patches3, out_patch4 = out_patches.split([n1, n2, n3, 1], 2)
        out_patches1 = F.fold(out_patches1, [self.patch_size * n3, self.patch_size * n2], self.patch_size, stride=self.patch_size)
        out_patches2 = F.fold(out_patches2, [self.patch_size, self.patch_size * n2], self.patch_size, stride=self.patch_size)
        out_patches3 = F.fold(out_patches3, [self.patch_size * n3, self.patch_size], self.patch_size, stride=self.patch_size)
        out_patch4 = out_patch4.reshape(N, C, self.patch_size, self.patch_size) 
        
        outp = x.new(x.shape)
        outp[:, :, :self.patch_size * n3, :self.patch_size * n2] = out_patches1
        outp[:, :, -self.patch_size:, :self.patch_size * n2] = out_patches2
        outp[:, :, :self.patch_size * n3, -self.patch_size:] = out_patches3
        outp[:, :, -self.patch_size:, -self.patch_size:] = out_patch4
        
        return outp
    
    def forward(self, x):
        if self.training:
            return self.__train_forward(x)
        else:
            return self.__infer_forward(x)
