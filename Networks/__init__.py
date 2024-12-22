from Networks.MOE_KD import Model as os_kd
from Networks.os import decoder as baseline

model_dict = {
    "os_kd":os_kd,
    "baseline":baseline,
}