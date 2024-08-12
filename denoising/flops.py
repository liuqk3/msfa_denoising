import torch
from models import *
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info

# # model = Pan_MSFA()
# # model = sqad().cpu()
# model = sert_small()
model = NoiseDecoupledNet_PosEmb(U_Net(25, 25), U_Net_pos_emb(25, 25, 25, 128, 128))
model = sst()
input_tensor = torch.rand(1, 25, 128, 128)
model(input_tensor)

tensor = (input_tensor,)
flops = FlopCountAnalysis(model, tensor)
print("FLOPs: ", flops.total()/1e9)

# model = sert_base()
macs, params = get_model_complexity_info(model, (25, 128, 128), print_per_layer_stat=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))