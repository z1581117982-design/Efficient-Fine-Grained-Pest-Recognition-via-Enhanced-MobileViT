import torch
from ptflops import get_model_complexity_info
from model_lib.model_cbam_DFc import mobile_vit_small as create_model

model = create_model(num_classes=102)
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, (3, 224, 224))
    print(f"FLOPs: {macs}, Params: {params}")
