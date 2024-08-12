import os
import cv2
from config_decouple import Config
# from denoising.models.Unet_HP import CF_Net_ablation
# from denoising.models.Unet_HP import CF_Net
# from denoising.models.UCTransNet.UCTransNet import UCTransNet
# from denoising.utils.model_utils import set_lr 
# opt = Config('training.yml')
opt = Config()

# gpus = ','.join([str(i) for i in opt.GPU])
gpus = '0'
# import pdb; pdb.set_trace()
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
from dataloaders.data_rgb import get_training_data, get_training_syn_data, get_validation_data, get_training_syn_data_cb_param, get_validation_data_cb_param, get_training_real_data_cb_param
from dataloaders.data_rgb import get_training_syn_data_coord
from pdb import set_trace as stx

from models import *
from losses import CharbonnierLoss

from tqdm import tqdm 
import tensorboardX
from tensorboardX import SummaryWriter
from datetime import datetime



criterion = nn.L1Loss()


######### Set Seeds ###########
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# set_seed(114514)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

writer = SummaryWriter(os.path.join(model_dir, 'log/log_decouple_grnet_subnet1'))

utils.mkdir(result_dir)
utils.mkdir(model_dir)

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
# model_restoration = UNet_double_cb_finetune(25, 25)
# model_restoration = UNet_double_cbv2_finetune(25, 25)
subnet1 = U_Net(25, 25)
subnet2 = U_Net_pos_emb(25, 25, 25, 217, 409)
# subnet2 = U_Net(25, 25)

device = "cuda:0"

# model_restoration.cuda()
subnet1.to(device)
subnet2.to(device)

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

######### DataLoaders ###########
img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}

train_dataset = get_training_syn_data_coord(data_dir, train_list, opt.MODEL.RATIO, opt.NOISEMODEL, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=True)

val_dataset = get_validation_data(data_dir, val_list, opt.MODEL.RATIO)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS))
print('===> Loading datasets')

print('Training dataset: '+str(len(train_loader)))
print('Validation dataset: '+str(len(val_loader)))
eval_now = len(train_loader) - 1
print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

optimizer = optim.Adam(subnet1.parameters(), lr=1e-4, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

best_psnr = 0
best_epoch = 0
n_iter = 0
# import pdb; pdb.set_trace()
if os.path.exists(os.path.join(model_dir, 'subnet1_latest.pth')):
    path_chk_rest = utils.get_last_path(model_dir, 'subnet1_latest.pth')
    utils.load_checkpoint(subnet1,path_chk_rest)
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

# train net1
for epoch in range(start_epoch, 250 + 1):
    subnet1.train()
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
        
    for i, data in enumerate(tqdm(train_loader), 0): 

        # zero_grad
        for param in subnet1.parameters():
            param.grad = None

        target = data[0].to(device)
        input_ = data[1].to(device)
        input_cb = data[2].to(device)

        restored = subnet1(input_)
        # restored = subnet1(input_.unsqueeze(1)).squeeze(1)
        loss = criterion(restored, input_cb)
    
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
        n_iter += 1

    writer.add_scalar('Train/Loss', epoch_loss, epoch)

    if epoch == 150:
        utils.model_utils.set_lr(optimizer, 1e-5)
    if epoch == 300:
        utils.model_utils.set_lr(optimizer, 1e-6)
    
    print("------------------------------------------------------------------")
    print("Subnet1 Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, optimizer.param_groups[0]['lr']))
    print("------------------------------------------------------------------")
    # it_total+=1

    writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)

    # if epoch % 5 == 0:
    torch.save({'epoch': epoch, 
                'state_dict': subnet1.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'n_iter' : n_iter,
                'best_epoch' : best_epoch,
                'best_psnr' : best_psnr
                }, os.path.join(model_dir,"subnet1_latest.pth"))  

    # break

start_epoch = 1

optimizer = optim.Adam(subnet2.parameters(), lr=1e-4, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

best_psnr = 0
best_epoch = 0
n_iter = 0
# import pdb; pdb.set_trace()
if os.path.exists(os.path.join(model_dir, 'subnet2_latest.pth')):
    path_chk_rest = utils.get_last_path(model_dir, 'subnet2_latest.pth')
    utils.load_checkpoint(subnet2,path_chk_rest)
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

writer = SummaryWriter(os.path.join(model_dir, 'log/log_decouple_grnet_subnet2'))

# train net2
for epoch in range(start_epoch, 250 + 1):
    subnet2.train()
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
        
    for i, data in enumerate(tqdm(train_loader), 0): 

        # zero_grad
        for param in subnet2.parameters():
            param.grad = None

        target = data[0].to(device)
        input_ = data[1].to(device)
        input_cb = data[2].to(device)
        h = data[3][0].item()
        w = data[3][1].item()

        restored = subnet2(input_cb, h, w)
        # restored = subnet2(input_cb)
        # restored = subnet2(input_cb.unsqueeze(1)).squeeze(1)
        loss = criterion(restored, target)
    
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
        n_iter += 1

    writer.add_scalar('Train/Loss', epoch_loss, epoch)

    if epoch == 150:
        utils.model_utils.set_lr(optimizer, 1e-5)
    if epoch == 300:
        utils.model_utils.set_lr(optimizer, 1e-6)
    
    print("------------------------------------------------------------------")
    print("Subnet2 Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, optimizer.param_groups[0]['lr']))
    print("------------------------------------------------------------------")
    # it_total+=1

    writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)

    # if epoch % 5 == 0:
    torch.save({'epoch': epoch, 
                'state_dict': subnet2.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'n_iter' : n_iter,
                'best_epoch' : best_epoch,
                'best_psnr' : best_psnr
                }, os.path.join(model_dir,"subnet2_latest.pth"))   


    # break

writer = SummaryWriter(os.path.join(model_dir, 'log/log_decouple_grnet_overall'))
overall_net = NoiseDecoupledNet_PosEmb(subnet1, subnet2)
# overall_net = NoiseDecoupledNet(subnet1, subnet2)
# overall_net.net1.load_state_dict(torch.load(os.path.join(model_dir, 'subnet1_latest.pth'))['state_dict'])
# overall_net.net2.load_state_dict(torch.load(os.path.join(model_dir, 'subnet2_latest.pth'))['state_dict'])
overall_net.to(device)
optimizer = optim.Adam(overall_net.parameters(), lr=1e-5, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

best_psnr = 0
best_epoch = 0
n_iter = 0

start_epoch = 1
# import pdb; pdb.set_trace()
if os.path.exists(os.path.join(model_dir, 'overall_net_latest.pth')):
    path_chk_rest = utils.get_last_path(model_dir, 'overall_net_latest.pth')
    utils.load_checkpoint(overall_net,path_chk_rest)
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

# finetune
for epoch in range(start_epoch, 100 + 1):
    overall_net.train()
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
        
    for i, data in enumerate(tqdm(train_loader), 0): 

        # zero_grad
        for param in overall_net.parameters():
            param.grad = None

        target = data[0].to(device)
        input_ = data[1].to(device)
        input_cb = data[2].to(device)

        h = data[3][0].item()
        w = data[3][1].item()

        _, restored = overall_net(input_, h, w)
        # _, restored = overall_net(input_)
        # _, restored = overall_net(input_.unsqueeze(1))
        # restored = restored.squeeze(1)
        loss = criterion(restored, target)
    
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
        n_iter += 1

    writer.add_scalar('Train/Loss', epoch_loss, epoch)

    if epoch == 50:
        utils.model_utils.set_lr(optimizer, 1e-6)

    
    print("------------------------------------------------------------------")
    print("overall_net Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, optimizer.param_groups[0]['lr']))
    print("------------------------------------------------------------------")
    # it_total+=1

    writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)

    # if epoch % 5 == 0:
    torch.save({'epoch': epoch, 
                'state_dict': overall_net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'n_iter' : n_iter,
                'best_epoch' : best_epoch,
                'best_psnr' : best_psnr
                }, os.path.join(model_dir,"overall_net_latest.pth"))   

    # break