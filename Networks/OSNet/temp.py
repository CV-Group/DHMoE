import torch
from torch import nn
import copy

from .teacher import load_osnet_weights
from .student import student_net 


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        decoder = nn.DataParallel(student_net())

        self.teacher_all = decoder.module

        self.teacher = self.teacher_all.vgg
        self.upsample_t = self.teacher_all.de_pred

        self.student = nn.Sequential(load_osnet_weights(), nn.Conv2d(256, 512, 3, 1, 1))
        self.upsample_s = copy.deepcopy(self.upsample_t)
        self.student_all = nn.Sequential(self.student, self.upsample_s)
    

    def frezze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_t = self.teacher(x)
        output_t = self.upsample_t(x_t)

        x_s = self.student(x)
        output_s = self.upsample_s(x_s)

        return x_t, x_s, output_t, output_s


if __name__ == "__main__":
    model = Model().cuda()

    input_data = torch.randn(1, 3, 256, 256).cuda()
    output = model(input_data)
