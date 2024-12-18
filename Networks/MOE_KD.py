import torch
from torch import nn
import copy
import torch.nn.functional as F
from .student import load_osnet_weights as student_net
from .teacher import decoder as teacher_net
from .teacher import load_osnet_weights as teacher

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = teacher()      
        tea = teach()
        self.teacher_all = tea
        self.upsample_t = self.teacher_all.de_pred
        self.upsample_s = copy.deepcopy(self.upsample_t)       
        self.student = student_net()

    def forward(self, x):
        x1,x2,x3,x4,x5 = self.teacher(x)
        s_out, experts = self.student(x)      
        s_out = self.upsample_s(s_out)
        return x1, x2, x3, x4, x5, s_out, experts
    
def teach():
    net = teacher_net()
    model_path = '/home/jingan/wangluo/counting-total/ablation/1.0/carpk/train/model_best.pth'
    state_dict = torch.load(model_path)['state_dict']
    state_dict1 = {}

    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k.replace('module.', '', 1)   
        else:
            new_key = k
        state_dict1[new_key] = v
    net.load_state_dict(state_dict1,strict=True)
    for param in net.parameters():
            param.requires_grad = False
    return net
    
if __name__ == "__main__":
    model = Model().cuda()
    input_data = torch.randn(1, 3, 256, 256).cuda()
    output = model(input_data)
