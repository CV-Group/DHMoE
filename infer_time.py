import torch
from Networks.MOE_KD import Model  
import torch.nn as nn
from thop import profile
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

iterations = 300   
model = Model().cuda() 
device = torch.device("cuda:0")
model.to(device)
random_input = torch.randn(1, 3, 256, 256).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

for _ in range(50):
    _ = model(random_input)

times = torch.zeros(iterations)   
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        times[iter] = curr_time

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
flops, params = profile(model, inputs=(random_input,))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')



