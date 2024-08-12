import torch
import numpy as np
import pickle
import cv2
# import matplotlib
# matplotlib.use('Agg')
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn.functional as F
from scipy.special import kl_div
from scipy.stats import entropy

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def is_tif_file(filename):
    return any(filename.endswith(extension) for extension in [".tif"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

def load_tif_img(filepath):
    img = io.imread(filepath)
    # print(img.shape)
    img = img[18:118]
    img = img[::3]
    img = img.astype(np.float32)
    img = img/4096.
    return img

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def MyPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    ps = 0.
    for k in range(tar_img.size(1)):
        rmse = (imdff[:,k]**2).mean().sqrt()
        ps += 20*torch.log10(1/rmse)
    return ps/tar_img.size(1)

def batch_PSNR(img1, img2, data_range=None):
    # img1 *= torch.mean(img2)/torch.mean(img1)
    PSNR = []
    for im1, im2 in zip(img1, img2):
        # psnr = myPSNR(im1, im2)
        im1 = im1.detach().cpu().numpy()
        im2 = im2.detach().cpu().numpy()
        im1 = np.transpose(im1, (1,2,0))
        im2 = np.transpose(im2, (1,2,0))        
        psnr = peak_signal_noise_ratio(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR)

def batch_mse(img1, img2):
    # return F.mse_loss(img1, img2).detach().cpu().numpy()
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    mse = np.mean((img1 - img2) ** 2)
    return mse


def batch_SSIM(img1, img2, data_range=None):
    SSIM = []

    for im1, im2 in zip(img1, img2):
        im1 = im1.detach().cpu().numpy()
        im2 = im2.detach().cpu().numpy()
        im1 = np.transpose(im1, (1,2,0))
        im2 = np.transpose(im2, (1,2,0))
        ssim = structural_similarity(im1, im2, multichannel=True)
        SSIM.append(ssim)

    return sum(SSIM)/len(SSIM)

def calculate_sam(img1, img2):
    pass

def compute_sam(x_true, x_pred):
    
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape

    w, h, c = x_true.shape
    x_true = x_true.reshape(-1, c)
    x_pred = x_pred.reshape(-1, c)

    x_pred[np.where((np.linalg.norm(x_pred, 2, 1))==0),]+=0.0001
    
    sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))

    sam = np.arccos(sam) * 180 / np.pi
    mSAM = sam.mean()
    # var_sam = np.var(sam)
    return mSAM

def compute_sam_v2(img1, img2):
    """SAM for 3D image, shape (H, W, C); uint or float[0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    inner_product = (img1_ * img2_).sum(axis=2)
    img1_spectral_norm = np.sqrt((img1_ ** 2).sum(axis=2))
    img2_spectral_norm = np.sqrt((img2_ ** 2).sum(axis=2))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0,
                                                                                                            max=1)
    return np.mean(np.arccos(cos_theta)) * 180 / np.pi

def batch_SAM(img1, img2, data_range=None):
    SAM = []
    for im1, im2 in zip(img1, img2):
        im1 = im1.detach().cpu().numpy()
        im2 = im2.detach().cpu().numpy()
        im1 = np.transpose(im1, (1,2,0))
        im2 = np.transpose(im2, (1,2,0))
        sam = compute_sam_v2(im1, im2)
        SAM.append(sam)

    return sum(SAM)/len(SAM)

def compute_ergas(img1, img2, scale=4):
    
    d = img1 - img2
    ergasroot = 0
    for i in range(d.shape[2]):
        ergasroot = ergasroot + np.mean(d[:,:,i]**2)/np.mean(img2[:,:,i])**2
    
    ergas = 100/scale*np.sqrt(ergasroot/d.shape[2])
    return ergas

def compute_ergas_v2(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_) ** 2)
        return 100 / scale * np.sqrt(mse / (mean_real ** 2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_) ** 2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real ** 2 + np.finfo(np.float64).eps)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')

def batch_ERGAS(img1, img2, data_range=None):
    ERGAS = []
    for im1, im2 in zip(img1, img2):
        im1 = im1.detach().cpu().numpy()
        im2 = im2.detach().cpu().numpy()
        im1 = np.transpose(im1, (1,2,0))
        im2 = np.transpose(im2, (1,2,0))
        ergas = compute_ergas_v2(im1, im2)
        ERGAS.append(ergas)

    return sum(ERGAS)/len(ERGAS)


def compute_kl_and_ce(noisy_img1, noisy_img2, clean_img, channel_wise=False):
    
    clean_img = clean_img.astype(np.float32)
    # import pdb; pdb.set_trace()
    noisy_img1 = noisy_img1.astype(np.float32) - clean_img
    pdf_img1 = noisy_img1 - np.min(noisy_img1) + 1e-8
    

    noisy_img2 = noisy_img2.astype(np.float32) - clean_img
    pdf_img2 = noisy_img2 - np.min(noisy_img2) + 1e-8
    

    if channel_wise:
        channels = pdf_img1.shape[0]
        kls = []
        ces = []
        for c in range(channels):
            pdf_img1_tmp = pdf_img1[c] / np.sum(pdf_img1[c])
            pdf_img2_tmp = pdf_img2[c] / np.sum(pdf_img2[c])
            kls.append(np.sum(kl_div(pdf_img1_tmp, pdf_img2_tmp)))
            ces.append(np.sum(entropy(pdf_img1_tmp, pdf_img2_tmp)))
        kl = np.mean(kls)
        ce = np.mean(ces)
    else:
        pdf_img1 = pdf_img1 / np.sum(pdf_img1)
        pdf_img2 = pdf_img2 / np.sum(pdf_img2)
        kl = np.sum(kl_div(pdf_img1, pdf_img2))
        ce = np.sum(entropy(pdf_img1, pdf_img2))

    return kl, ce