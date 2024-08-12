

"""
## Learning Enriched Features for Real Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## ECCV 2020
## https://arxiv.org/abs/2003.06792
"""


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
import cv2
# from networks.MIRNet_model import MIRNet
from models import *
from dataloaders.data_rgb import get_validation_data, get_training_data
import utils
from skimage import img_as_ubyte
import scipy.io as scio
import h5py
from os.path import join

from PIL import Image

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--data_dir', default='/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v4',
    type=str, help='Directory of validation images')
parser.add_argument('--test_list', default='/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/dataset_v4/testval.txt')
parser.add_argument('--ratio', default=10, type=int, help='Ratio')
parser.add_argument('--result_dir', default='./results/denoising/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='checkpoints/global/models/ps128_bs1/model_best.pth',
    type=str, help='Path to weights')# Synthetic
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--imgpath', default='', type=str, help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.data_dir, args.test_list, args.ratio)
# test_dataset = get_training_data(args.input_dir, args.ratio)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)

model_restoration = CF_Net(25, 25, 32, 8)

utils.load_checkpoint(model_restoration,args.weights)
# checkpoint = torch.load(args.weights)
# model_restoration.load_state_dict(checkpoint['net'])
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()

# model_restoration=nn.DataParallel(model_restoration)

model_restoration.eval()

with torch.no_grad():
    psnr_val_rgb = []
    sam_val = []
    ssim_val = []
    ergas_val = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):

        # if ii not in [12, 19]:
        #     continue

        rgb_gt = data_test[0].cuda()
        rgb_noisy = data_test[1].cuda()
        filenames = data_test[2]
        rgb_restored,_ = model_restoration(rgb_noisy)
        # rgb_restored = model_restoration(rgb_noisy)
        # rgb_restored = rgb_noisy
        # real_hsi = h5py.File('../QRNN3D/matlab/Result/Urban/Urban_F210/None.mat', 'r')
        # real_hsi = real_hsi['R_hsi'][:].transpose((0,2,1)).astype(np.float32)
        # c,h,w = real_hsi.shape
        # rgb_noisy = torch.from_numpy(real_hsi[:130,:h//16*16-32,:w//16*16-32]).cuda().unsqueeze(0)
        # rgb_restored = model_restoration(rgb_noisy.unsqueeze(1))[:,0]
        rgb_restored = torch.clamp(rgb_restored,0,1)
        # rgb_restored = rgb_noisy
     
        # psnr_val_rgb.append(utils.batch_PSNR(rgb_restored[:,30], rgb_gt[:,30], 1.))
        # psnr_val_rgb.append(utils.batch_PSNR(rgb_noisy[:,30], rgb_gt[:,30], 1.))

        psnr_val_rgb.append(utils.batch_PSNR(rgb_restored, rgb_gt, 1.))
        sam_val.append(utils.batch_SAM(rgb_restored, rgb_gt, 1.))
        ssim_val.append(utils.batch_SSIM(rgb_restored, rgb_gt, 1.))
        ergas_val.append(utils.batch_ERGAS(rgb_restored, rgb_gt, 1.))
        # psnr_val_rgb.append(utils.batch_PSNR(rgb_noisy, rgb_gt, 1.))

        if args.imgpath != '':
            if not os.path.exists(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing', args.imgpath, str(args.ratio), str(ii))):
                os.makedirs(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing', args.imgpath, str(args.ratio), str(ii)))
            rgb_restored = rgb_restored.cpu().detach().numpy()
            for batch in range(rgb_restored.shape[0]):
                # rgb_restored shape [B, C, H, W], range in [0,1]
                image = rgb_restored[batch]
                for k in range(25):
                    img = (image[k]*255).astype(np.uint8)
                    Image.fromarray(img).save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing', args.imgpath, str(args.ratio), str(ii), 'band_'+str(k)+'.png'))
                
                

        # rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
        # rgb_gt = rgb_gt.cpu().detach().numpy()
        # rgb_noisy = rgb_noisy.cpu().detach().numpy()
        # rgb_restored = rgb_restored.cpu().detach().numpy()

        # cv2.imshow('0', rgb_restored[0,106])
        # cv2.imwrite('Urban.png', rgb_restored[0,106]*255)
        # cv2.waitKey()

        # # rgb_noisy *= np.mean(rgb_gt)/np.mean(rgb_noisy)
        # rgb_restored *= np.mean(rgb_gt)/np.mean(rgb_restored)

        # for i in range(len(rgb_gt)):
        #     for j in range(34):
        #         cv2.imshow('gt', np.rot90(rgb_gt[i,j,::-1]))
        #         cv2.imshow('input', np.rot90(rgb_noisy[i,j,::-1]))
        #         cv2.imshow('output', np.rot90(rgb_restored[i,j,::-1]))
        #         cv2.waitKey()

        # if args.save_images:
        #     for batch in range(len(rgb_gt)):
        #         # denoised_img = img_as_ubyte(rgb_restored[batch])
        #         # utils.save_img(args.result_dir + filenames[batch][:-4] + '.png', denoised_img)
        #         denoised_img = rgb_restored[batch, 30, ::-1]
        #         # cv2.imwrite(args.result_dir + filenames[batch][:-4] + '.png', np.rot90(denoised_img)*255)
        #         denoised_hsi = np.rot90(rgb_restored[batch, :, ::-1], axes=(-2,-1))
        #         sio.savemat(args.result_dir + filenames[batch][:-4] + '.mat', {'R_hsi': np.transpose(denoised_hsi, (1,2,0))})

        #         # noisy_hsi = np.rot90(rgb_restored[batch, :, ::-1], axes=(-2,-1))
        #         # sio.savemat(args.result_dir + filenames[batch][:-4] + '.mat', {'noisy_hsi': np.transpose(noisy_hsi, (1,2,0))})

        #         # if 'Real' in args.result_dir:
        #         #     noisy = rgb_noisy[batch, 30, ::-1]
        #         #     cv2.imwrite(args.result_dir.replace('Real', 'input') + filenames[batch][:-4] + '.png', np.rot90(noisy*args.ratio)*255)

        #         #     gt = rgb_gt[batch, 30, ::-1]
        #         #     cv2.imwrite(args.result_dir.replace('Real{}'.format(args.ratio), 'gt') + filenames[batch][:-4] + '.png', np.rot90(gt)*255)
            
for i in range(len(psnr_val_rgb)):
    print('idx: %d, PSNR: %.4f, SSIM: %.4f, SAM: %.4f, ERGAS: %.4f' % (i, psnr_val_rgb[i], ssim_val[i], sam_val[i], ergas_val[i]))
# for i in range(len(psnr_val_rgb)):
#     print('idx: %d, PSNR: %.4f, SSIM: %.4f, SAM: %.4f' % (i, psnr_val_rgb[i], ssim_val[i], sam_val[i]))
psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
ssim_mean = sum(ssim_val)/len(ssim_val)
sam_mean = sum(sam_val)/len(sam_val)
ergas_mean = sum(ergas_val)/len(ergas_val)
print('PSNR: %.4f, SSIM: %.4f, SAM: %.4f, ERGAS: %.4f' % (psnr_val_rgb, ssim_mean, sam_mean, ergas_mean))

# print('PSNR: %.4f, SSIM: %.4f, SAM: %.4f' % (psnr_val_rgb, ssim_mean, sam_mean))