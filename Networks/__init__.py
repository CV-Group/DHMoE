from Networks.MOE_KD import Model as os_kd
from Networks.student import load_osnet_weights as os_stu
from Networks.OSNet.kd_moe_ablation.MOE import student_net as student_moe

model_dict = {
    "os_kd":os_kd,
    "os_stu":os_stu,
    "student_moe":student_moe,
}