import torch
from .osnet import load_osnet_weights
from .teacher import decoder as Decoder
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        decoder = nn.DataParallel(Decoder())
        decoder.load_state_dict(
            torch.load(
                "/home/wenzhe/Experiments/heu/Quantum/Counting-Trans/ablation/shanghaiA_osnet/train/model_best.pth"
            )["state_dict"]
        )
        for param in decoder.parameters():
            param.requires_grad = False

        teacher = list(decoder.module.vgg.children())
        self.upsample = decoder.module.de_pred
        
        self.block1 = nn.Sequential(*teacher[:2])
        self.block2 = teacher[2]
        self.block3 = teacher[3]
        self.block4 = teacher[4]
        self.block5 = teacher[5]
        
        self.teacher = nn.ModuleList(decoder.module.vgg.children())
        
        self.align_conv = nn.ModuleList([
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(192, 384, 3, 1, 1), 
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1)
        ])
        self.student = nn.ModuleList(load_osnet_weights().children())
        self.student1 = nn.Sequential(load_osnet_weights(), nn.Conv2d(256, 512, 3, 1, 1))
        # self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float))
        # self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float))
        
        # nn.Softmax()

    def forward(self, x):
        student = self.student1(x)
        x_ts = []  
        x_ss = []
        x_t = x
        for i in range(6):
            x_t = self.teacher[i](x_t)
            if i != 1:
                x_ts.append(x_t)
            
            x = self.student[i](x)
            if i != 1:
                if i <1:
                    x_s = self.align_conv[i](x)
                else:
                    x_s = self.align_conv[i-1](x)
                    
                x_ss.append(x_s)
        
        output = self.upsample(student)
        
        return x_ts, x_ss, output 


if __name__ == "__main__":
    model = Model().cuda()

    input_data = torch.randn(1, 3, 256, 256).cuda()
    output = model(input_data) 
