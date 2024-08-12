import os
import cv2
from config import Config
# from denoising.models.Unet_HP import CF_Net_ablation
# from denoising.models.Unet_HP import CF_Net
# from denoising.models.UCTransNet.UCTransNet import UCTransNet
# from denoising.utils.model_utils import set_lr 
# opt = Config('training.yml')
opt = Config()

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from scipy import stats

import utils
from dataloaders.data_rgb import get_training_data, get_training_syn_data, get_validation_data
from dataloaders.data_rgb import get_training_syn_data_coord
from pdb import set_trace as stx

from models import *
from losses import CharbonnierLoss

from tqdm import tqdm 
import tensorboardX
from tensorboardX import SummaryWriter
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--model', default='unet',
    type=str, help='select model')
parser.add_argument('--savepath',type=str,help='path to save checkpoint')
args = parser.parse_args()


if opt.DATA == 'syn':
    writer = SummaryWriter('log/log_'+opt.DATA+'_'+str(opt.MODEL.RATIO)+'_'+opt.NOISEMODEL+'_'+datetime.now().strftime("%Y%m%d%H%M%S"))
else:
    writer = SummaryWriter('log/log_'+opt.DATA+'_'+str(opt.MODEL.RATIO)+'_'+datetime.now().strftime("%Y%m%d%H%M%S"))

criterion = nn.L1Loss()

def fft_loss(output, target, criterion):
    output = torch.rfft(output, signal_ndim=2, normalized=False, onesided=False)
    target = torch.rfft(target, signal_ndim=2, normalized=False, onesided=False)

    return criterion(output, target)


def density_loss(restored, target, epoch):
    target_density = density(target)
    # ratio = min(np.exp(epoch-175), 1)
    # ratio = min((epoch-100)/, 1)
    ratio = max(0, (epoch-100)/1500)
    ratio = min(ratio, 0.1)
    loss = torch.mean(torch.abs(restored-target)*(1+ratio*target_density))
    # loss = criterion(restored,target) + torch.mean(torch.abs((restored-target)*ratio*target_density))
    
    return loss


def density(x):
    x = torch.clamp(x*255,0.,255.).detach()
    # x = torch.mean(x, dim=1)
    b,c,w,h = x.shape
    
    im_sum = []
    for i in range(b):
        im_sum_channel = []
        for j in range(c):
            im = np.array(x[i, j].cpu()).astype(np.uint8)
            im_blur = cv2.GaussianBlur(im, (5,5), 0)
            im_minus = abs(im.astype(np.float) - im_blur.astype(np.float)).astype(np.uint8)
            im_edge = cv2.GaussianBlur(im_minus, (5,5), 0).astype(np.float)
            im_edge = (im_edge - np.min(im_edge)) / (np.max(im_edge) - np.min(im_edge))
            im_edge = torch.from_numpy(im_edge)
            im_sum_channel.append(im_edge.unsqueeze(0))
        im_sum.append(torch.cat(im_sum_channel, dim=0).unsqueeze(0))

    im_sum = torch.cat(im_sum, dim=0).float().cuda()
    
    return im_sum


######### Set Seeds ###########
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
# random.seed(1234)
# np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)

# set_seed(123)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

# result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)
# model_dir = os.path.join(args.savepath, args.model, 'no_decouple')
# utils.mkdir(result_dir)
utils.mkdir(model_dir)

# train_dir = opt.TRAINING.TRAIN_DIR
# val_dir   = opt.TRAINING.VAL_DIR
data_dir = opt.TRAINING.DATA_DIR
train_list = opt.TRAINING.TRAIN_FILE
val_list = opt.TRAINING.VAL_FILE
save_images = opt.TRAINING.SAVE_IMAGES

######### Model ###########
# model_restoration = U_Net_GR(25, 25)
# model_restoration = MIRNet()
# model_restoration = U_Net(25, 25)
# model_restoration = GRNet(25, 25)
# model_restoration = UCTransNet(25, 25, 128)
# model_restoration = U_Net_3D()
# model_restoration = U_Net_3D_QRU()
# model_restoration = HSIDCNN(5)
# model_restoration = qrnn3d()
# model_restoration = SQADNet()
# model_restoration = U_Net_HP(25, 25, 8)
# model_restoration = CF_Net_edge_v3(25, 25, 64)
# model_restoration = U_Net_edge_v2(25, 25)
# model_restoration = Pan_MSFA()
# model_restoration = U_Net_HP_Final(25, 25, 8)
# model_restoration = GRNet(25)
# model_restoration = HSIDAN(5)
# model_restoration = sqad()
# model_restoration = U_Net_FFC(25, 25)
# model_restoration = U_Net(25, 25)
# model_restoration = NoiseDecoupledNet_PosEmb(U_Net(25, 25), U_Net_pos_emb(25, 25, 25, 217, 409))
# model_restoration = sert_tiny()
# model_restoration = sert_base()
# model_restoration = UNet_double_finetune(25, 25)
# model_restoration = U_Net_GF(25, 25)
if args.model in ['GRNet', 'sert']:
    if args.model == 'GRNet':
        subnet1 = GRNet(25)
        subnet2 = GRNet(25)
    elif args.model == 'sert':
        subnet1 = sert_base()
        subnet2 = sert_base()
    model_restoration = NoiseDecoupledNet(subnet1, subnet2)
model_restoration = SCUNet()
# utils.load_checkpoint(model_restoration,'./checkpoints/Denoising/models/Synthetic50_3D/model_best.pth')
# utils.load_checkpoint(model_restoration,'./checkpoints/Denoising/models/QRNN3D50_S/model_best.pth')
# checkpoint = torch.load('./checkpoints/Denoising/models/qrnn3d/complex/model_best.pth')
# model_restoration.load_state_dict(checkpoint['net'])



# model_restoration.cuda()
model_restoration.to("cuda:0")

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [150, 180], 0.1)

mixup = utils.MixUp_AUG()
best_psnr = 0
best_epoch = 0
n_iter = 0

######### Resume ###########
if opt.TRAINING.RESUME:
    # path_chk_rest = utils.get_last_path(model_dir, '_295.pth')
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

    ckpt = torch.load(path_chk_rest)
    best_psnr = ckpt['best_psnr']
    best_epoch = ckpt['best_epoch']
    n_iter = ckpt['n_iter']

    for p in optimizer.param_groups: p['lr'] = lr
    warmup = False
    new_lr = lr
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:",new_lr)
    print('------------------------------------------------------------------------------')
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 180], 0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-start_epoch+1, eta_min=1e-6)
# else:
#     warmup = True

######### Scheduler ###########
# if warmup:
#     warmup_epochs = 3
#     scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=1e-6)
#     scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
#     scheduler.step()

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

######### Loss ###########
# criterion = CharbonnierLoss().cuda()


######### DataLoaders ###########
img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}

if opt.DATA == 'real':
    train_dataset = get_training_data(data_dir, train_list, opt.MODEL.RATIO, img_options_train)
else:
    train_dataset = get_training_syn_data_coord(data_dir, train_list, opt.MODEL.RATIO, opt.NOISEMODEL, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=True)

# val_data_dir = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/syn_data_cb'
# val_data_dir = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/syn_data_zero_mean'
# val_dataset = get_validation_data(val_data_dir, val_list, opt.MODEL.RATIO)
val_dataset = get_validation_data(data_dir, val_list, opt.MODEL.RATIO)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS))
print('===> Loading datasets')

print('Training dataset: '+str(len(train_loader)))
print('Validation dataset: '+str(len(val_loader)))
eval_now = len(train_loader) - 1
print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

# scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    model_restoration.train()
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
        
    for i, data in enumerate(tqdm(train_loader), 0): 
        # print(stats.entropy(np.abs(data[1].numpy().flatten())+1e-3, np.abs(data[0].numpy().flatten())+1e-3))

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()
        input_cb = data[2].cuda()
        h = data[3][0].item()
        w = data[3][1].item()

        # target_density = density(target)

        # if epoch>5:
        #     target, input_ = mixup.aug(target, input_)
        # _, restored = model_restoration(input_)
        restored = model_restoration(input_)
        # restored1, restored2 = model_restoration(input_, h, w)
        # restored = model_restoration(input_.unsqueeze(1)).squeeze(1)
        # restored = torch.clamp(restored,0,1)  
        # loss = criterion(restored2, target) + criterion(restored1, input_cb)
        loss = criterion(restored, target)
        # density(target)
        # loss = torch.mean(torch.abs(restored-target)*(1+target_density))
        # loss = density_loss(restored, target, epoch)
        # loss = torch.mean(torch.abs(restored-target)*(1+0.1*target_density))

        # fine, coarse = model_restoration(input_)
        # fine = torch.clamp(fine, 0, 1)
        # coarse = torch.clamp(coarse, 0, 1)
        # loss = criterion(fine, target) + 0.1 * criterion(coarse, target)
    
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
        n_iter += 1

        # break


        # writer.add_scalar('Train/Loss', loss.item(), n_iter)
    writer.add_scalar('Train/Loss', epoch_loss, epoch)

    #### Evaluation ####
    # if save_images:
    #     utils.mkdir(result_dir + '%d/%d'%(epoch,i))
    # model_restoration.eval()
    # with torch.no_grad():
    #     psnr_val_rgb = []
    #     for ii, data_val in enumerate((val_loader), 0):
    #         target = data_val[0].cuda()
    #         input_ = data_val[1].cuda()
    #         filenames = data_val[2]

    #         # import pdb; pdb.set_trace()

    #         restored = model_restoration(input_)
    #         # restored = model_restoration(input_.unsqueeze(1)).squeeze(1)
    #         restored = torch.clamp(restored,0,1)
    #         # psnr_val_rgb.append(utils.batch_PSNR(restored.squeeze(1), target, 1.))
    #         psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.))

    #         if save_images:
    #             target = target.permute(0, 2, 3, 1).cpu().detach().numpy()
    #             input_ = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
    #             restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                
    #             for batch in range(input_.shape[0]):
    #                 temp = np.concatenate((input_[batch]*255, restored[batch]*255, target[batch]*255),axis=1)
    #                 utils.save_img(os.path.join(result_dir, str(epoch), str(i), filenames[batch][:-4] +'.jpg'),temp.astype(np.uint8))

    #     psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
        
    #     if psnr_val_rgb > best_psnr:
    #         best_psnr = psnr_val_rgb
    #         best_epoch = epoch
    #         # best_iter = i 
    #         torch.save({'epoch': epoch, 
    #                     'state_dict': model_restoration.state_dict(),
    #                     'optimizer' : optimizer.state_dict()
    #                     }, os.path.join(model_dir,"model_best.pth"))

    #     print("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_psnr))
        
    # writer.add_scalar('Val/PSNR', psnr_val_rgb, epoch)

    # scheduler.step()
    if epoch == 150:
        utils.model_utils.set_lr(optimizer, 1e-5)
    if epoch == 300:
        utils.model_utils.set_lr(optimizer, 1e-6)
    
    print("------------------------------------------------------------------")
    # print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\t".format(epoch, time.time()-epoch_start_time,epoch_loss))
    # print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, optimizer.param_groups[0]['lr']))
    print("------------------------------------------------------------------")
    # it_total+=1

    writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)

    # if epoch % 5 == 0:
    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'n_iter' : n_iter,
                'best_epoch' : best_epoch,
                'best_psnr' : best_psnr
                }, os.path.join(model_dir,"model_latest.pth"))   

        # torch.save({'epoch': epoch, 
        #             'state_dict': model_restoration.state_dict(),
        #             'optimizer' : optimizer.state_dict(),
        #             'n_iter' : n_iter,
        #             'best_epoch' : best_epoch,
        #             'best_psnr' : best_psnr
        #             }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

