import torch
import sys
sys.path.append("..")
from model_lib.model_ECANet import mobile_vit_small
# 如果你想测试其他模型，也可以改成：
# from model_SimAM_DFc_ECANet import mobile_vit_x_small
# from model_SimAM_DFc_ECANet import mobile_vit_xx_small

def count_parameters_in_m(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params / 1e6:.3f} M")
    print(f"Trainable params: {trainable_params / 1e6:.3f} M")

if __name__ == "__main__":
    model = mobile_vit_small(num_classes=102)   # 按你的类别数修改
    count_parameters_in_m(model)