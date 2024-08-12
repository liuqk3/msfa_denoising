from .msesr import *
from .dncnn import *
from .rednet import *
from .Unet_modified import *
from .Unet3D import *
from .Unet3D_QRU import *
from .qrnn import QRNNREDC3D
from .HSIDCNN import HSIDCNN
# from .UCTransNet import *
from .GRNet import *
from .UNetG import *
# from .HSIDAN import *
from .SQAD import sqad
from .Unet_HP import U_Net_HP
from .CFNet_edge import CF_Net, CF_Net_edge_v1, CF_Net_edge_v2, CF_Net_edge_v3
from .Unet_edge_guided import U_Net_edge, U_Net_edge_v2, U_Net_edge_v3
from .Pan_MSFA import Pan_MSFA
from .Unet_ffc import U_Net_FFC, U_Net_GF
from .Unet_mod import U_Net_Mod
from .Unet_decouple import UNet_double, UNet_double_finetune, U_Net_cb, UNet_double_cb_finetune, U_Net_cbv2, U_Net_cbv2_norm, \
     UNet_double_cbv2_finetune, UNet_double_cbv2_joint, NoiseDecoupledNet, U_Net_pos_emb, NoiseDecoupledNet_PosEmb
from .Unet_adaptive import Ada_U_Net
from .Unet_pac import U_Net_pac
from .Unet_param import U_Net_param
from .SERT import sert_base, sert_real, sert_small, sert_tiny
from .SST import sst
from .SCUNet import SCUNet
def qrnn3d():
    # net = QRNNREDC3D(1, 16, 5, [1,3], has_ad=True, act='relu')
    net = QRNNREDC3D(1, 16, 5, [1,3], has_ad=True, act='tanh')
    net.use_2dconv = False
    net.bandwise = False
    return net
