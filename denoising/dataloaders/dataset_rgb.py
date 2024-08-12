from cmath import nan
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_tif_file, load_tif_img, Augment_RGB_torch, get_data_list
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

import torch.nn.functional as F
import random
import scipy.stats as stats
# from NoiseModel import NoiseModelV4

# augment   = Augment_RGB_torch()
# transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

def read_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        npArray = np.frombuffer(data, dtype='<f4')
        npArray = npArray
        # npArray = np.clip(npArray, 0, 1)
        npArray.shape = (1085, 2045)

    img = np.zeros((25, 217, 409), dtype=np.float32)
    
    for i in range(25):
        pattern_r = int(i/5)
        pattern_c = i % 5
        img[i, :, :] = npArray[pattern_c:1085:5, pattern_r:2045:5]
    # show_img(img-32)
    # img = (npArray-32) / (1023-32)
    img = (img - 32) / (1023 - 32)
    img = img * 1023
    # img = img / 1023
    # img = np.clip(img, 0, 1023)
    # print(np.mean(img))
    # print(np.max(img))
    return img

def linear(x, a, b):
    return a*x+b

def logarithm(x, a, b):
    # if b*x + c < 0:
    #     print(b*x+c)
    return a * np.log(x) + b

def quadra(x, a, b, c):

    return a*x*x + b*x + c

def linear_exp(x, a, b, c):
    return a * np.exp(b*x) + c

class NoiseModel:
    def __init__(self, model='PGBRC'):
        super().__init__()

        # self.param_dir = '../calibration/camera_params'#os.path.join('camera_params', 'V4')

        self.param_dir = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/camera_params'

        # if 'B' in model:
        if True:
            self.param_dir = self.param_dir + '_withcb'

        print('[i] NoiseModel with {}'.format(self.param_dir))
        print('[i] using noise model {}'.format(model))
        
        self.camera_params = {}
        self.camera_K = {}
 
        for band in range(25):
            self.camera_params[band] = np.load(os.path.join(self.param_dir, 'band_{}_params.npy'.format(band)), allow_pickle=True).item()
            
            if '_withcb' in self.param_dir:
            # if True:
                self.camera_K[band] = np.load(os.path.join(self.param_dir.replace('camera_params_withcb', 'K'), 'color_{}.npy'.format(band)), allow_pickle=True).item()
            else:
                self.camera_K[band] = np.load(os.path.join(self.param_dir.replace('camera_params', 'K'), 'color_{}.npy'.format(band)), allow_pickle=True).item()

        self.model = model

        self.defects_mask = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v4/final_mask.npy')

        self.biasframe = []

        if 'F' in self.model:
            bf_path = '/home/jiangyuqi/Desktop/HSI_data/biasframe_1k'
            for d in os.listdir(bf_path):
                bf_p = os.path.join(bf_path, d, 'unprocessed', 'frame.raw')
                bf_data = read_file(bf_p)
                if np.mean(bf_data) < 3:
                    self.biasframe.append(bf_data)
                # self.biasframe.append(bf_data)

        # self.cb_log = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v2/logarithm.npy', allow_pickle=True).item()
        # self.cb_quadra = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias/quadra.npy', allow_pickle=True).item()
        # self.cb_exp = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias/exp.npy', allow_pickle=True).item()

        # self.color_bias_a = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_log/a_matrix.npy')
        # self.color_bias_b = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_log/b_matrix.npy')
        self.color_bias_a = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_exp/a_matrix.npy')
        self.color_bias_b = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_exp/b_matrix.npy')
        self.color_bias_c = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_exp/c_matrix.npy')
        # self.cb_param = {}
        # for band in range(25):
        #     self.cb_param[band] = np.load(os.path.join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/camera_params', 'band_{}_params.npy'.format(band)), allow_pickle=True).item()
        #     print(self.cb_param[band]['color_bias'])

        # self.cb_bias_scale = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_exp/scale_mean.npy')
        self.cb_bias_scale = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_exp/cb_gau_scale_v2.npy', allow_pickle=True).item()

    def _sample_params(self, band):
        Q_step = 1

        saturation_level = 1023
        profiles = ['Profile-1']

        camera_params = self.camera_params[band]
        camera_K = self.camera_K[band]

        if 'color_bias' in camera_params.keys():
        # if True:
            color_bias = np.random.choice(camera_params['color_bias'],1)
            # color_bias = np.random.choice(self.cb_param[band]['color_bias'])
        else:
            color_bias = None

        G_shape = np.random.choice(camera_params['G_shape'])
        profile = np.random.choice(profiles)
        camera_params = camera_params[profile]

        # log_K = np.random.uniform(low=np.log(Kmin), high=np.log(Kmax))
        # log_K = np.random.uniform(low=np.log(1e-1), high=np.log(30))
        log_K = np.log(camera_K['K'])
        
        log_g_scale = np.random.standard_normal() * camera_params['g_scale']['sigma'] * 1 +\
             camera_params['g_scale']['slope'] * log_K + camera_params['g_scale']['bias']
        
        log_G_scale = np.random.standard_normal() * camera_params['G_scale']['sigma'] * 1 +\
             camera_params['G_scale']['slope'] * log_K + camera_params['G_scale']['bias']

        log_R_scale = np.random.standard_normal() * camera_params['R_scale']['sigma'] * 1 +\
             camera_params['R_scale']['slope'] * log_K + camera_params['R_scale']['bias']

        log_C_scale = np.random.standard_normal() * camera_params['C_scale']['sigma'] * 1 +\
             camera_params['C_scale']['slope'] * log_K + camera_params['C_scale']['bias']

        K = np.exp(log_K)
        g_scale = np.exp(log_g_scale)
        G_scale = np.exp(log_G_scale)
        R_scale = np.exp(log_R_scale)
        C_scale = np.exp(log_C_scale)

        # color_bias = np.random.choice(camera_params['color_bias'])

        # ratio = 10 #np.random.uniform(low=100, high=300)
        # ratio = np.random.uniform(low=1, high=300)


        return np.array([K, g_scale, G_scale, G_shape, R_scale, C_scale, Q_step, saturation_level, color_bias], dtype=object) 

    def __call__(self, input, ratio=10, params=None):
        output = []

        if 'B' in self.model:
            bias_num = np.random.randn()

        for idx, band in enumerate(range(25)):
            y = input[idx:idx+1]
            if params is None:
                K, g_scale, G_scale, G_shape, R_scale, C_scale, Q_step, saturation_level, color_bias = self._sample_params(band)
            else:
                K, g_scale, G_scale, G_shape, R_scale, C_scale, Q_step, saturation_level, color_bias = params

            y = y * saturation_level
            y = y / ratio
            
            if 'P' in self.model:
                z = np.random.poisson(y / K).astype(np.float32) * K
            elif 'p' in self.model:
                z = y + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(K * y, 1e-10))
            else:
                z = y

            if 'b' in self.model:
                z = self.add_color_bias_v1(z, color_bias, ratio)
                # z = self.add_color_bias_v0(z, color_bias, self.cb_log[band], self.cb_quadra['ab'][band], self.cb_quadra['ac'][band])
            elif 'B' in self.model:
                # z = self.add_color_bias_v2(z, np.mean(y), self.cb_log[band], self.cb_quadra['ab'][band], self.cb_quadra['ac'][band])
                # z = self.add_color_bias_v3(z, self.color_bias_a[idx:idx+1], self.color_bias_b[idx:idx+1],self.color_bias_c[idx:idx+1], self.cb_bias_scale[band])
                z, cb = self.add_color_bias_v4(z, self.color_bias_a[idx:idx+1], self.color_bias_b[idx:idx+1],self.color_bias_c[idx:idx+1], band, bias_num)

            if 'F' in self.model:
                bf_index = np.random.choice(range(len(self.biasframe)), 1)
                biasframe = self.biasframe[bf_index[0]][band]
                z = z + biasframe

            if 'g' in self.model:
                z = z + np.random.randn(*y.shape).astype(np.float32) * np.maximum(g_scale, 1e-10) # Gaussian noise            
            # elif 'G' in self.model:
            #     z = z + stats.tukeylambda.rvs(G_shape, loc=0, scale=G_scale, size=y.shape).astype(np.float32) # Tukey Lambda 

            if 'R' in self.model:
                z = self.add_banding_noise(z, scale=R_scale, RC='R')
            
            if 'C' in self.model:
                z = self.add_banding_noise(z, scale=C_scale, RC='C')

            if 'U' in self.model:
                # z = z + np.random.uniform(low=-0.5*Q_step, high=0.5*Q_step)     
                z = np.round(z)

            if self.model == 'B':
                z = y + cb

            z = z * ratio
            z = z / saturation_level
            output.append(z)

        output = np.concatenate(output)

        if 'D' in self.model:
            for m in self.defects_mask:
                k, i, j = m
                output[k, i, j] = 1

        # return np.concatenate(output)
        return output

    def add_color_bias_v0(self, img, color_bias, log_parameter, quadra_parameter_ab, quadra_parameter_ac):

        q_ab_slope = quadra_parameter_ab['slope']
        q_ab_intercept = quadra_parameter_ab['intercept']
        q_ac_slope = quadra_parameter_ac['slope']
        q_ac_intercept = quadra_parameter_ac['intercept']
        mean_cb = color_bias
        sum_square = (img.shape[2]-1)*img.shape[2]*(2*img.shape[2]-1)/6/40/40
        sum_1st = (img.shape[2]-1)*img.shape[2]/2/40

        a = mean_cb * img.shape[2] - q_ab_intercept*sum_1st - q_ac_intercept*img.shape[2]
        a = a / (sum_square+q_ab_slope*sum_1st+q_ac_slope*img.shape[2])
        b = a * q_ab_slope + q_ab_intercept
        c = a * q_ac_slope + q_ac_intercept
        cb = np.array(range(img.shape[2])).astype(np.float32) / 40
        cb = quadra(cb, a, b, c)
        cb = cb.reshape(1, img.shape[2])
        img = img + cb

        return img

    def add_color_bias_v1(self, img, color_bias, ratio):
        img = img + color_bias
        return img

    def add_color_bias_v2(self, img, mean_val, log_parameter, quadra_parameter_ab, quadra_parameter_ac):
        #img shape: (1,217,409)

        q_ab_slope = quadra_parameter_ab['slope']
        q_ab_intercept = quadra_parameter_ab['intercept']
        q_ac_slope = quadra_parameter_ac['slope']
        q_ac_intercept = quadra_parameter_ac['intercept']

        log_scale = log_parameter['scale']
        log_slope = log_parameter['slope']
        # exp_intercept = log_parameter['intercept']

        mean_cb = logarithm(mean_val/1023, log_scale, log_slope) * 4
        # mean_cb = linear_exp(mean_val/1023, exp_scale, exp_slope, exp_intercept) * 4

        sum_square = (img.shape[2]-1)*img.shape[2]*(2*img.shape[2]-1)/6/40/40
        sum_1st = (img.shape[2]-1)*img.shape[2]/2/40

        a = mean_cb * img.shape[2] - q_ab_intercept*sum_1st - q_ac_intercept*img.shape[2]
        a = a / (sum_square+q_ab_slope*sum_1st+q_ac_slope*img.shape[2])
        b = a * q_ab_slope + q_ab_intercept
        c = a * q_ac_slope + q_ac_intercept
        cb = np.array(range(img.shape[2])).astype(np.float32) / 40
        cb = quadra(cb, a, b, c)
        cb = cb.reshape(1, img.shape[2])
        img = img + cb

        return img

    def add_color_bias_v3(self, img, parameter_a, parameter_b, parameter_c, bias_scale):
        bias = np.random.normal(scale=0.3)
        a_matrix = np.expand_dims(parameter_a, axis=1)
        a_matrix = np.repeat(a_matrix, img.shape[1], axis=1)
        b_matrix = np.expand_dims(parameter_b, axis=1)
        b_matrix = np.repeat(b_matrix, img.shape[1], axis=1)
        c_matrix = np.expand_dims(parameter_c, axis=1)
        c_matrix = np.repeat(c_matrix, img.shape[1], axis=1)
        # color_bias = logarithm(img/10, a_matrix, b_matrix) * 4
        color_bias = linear_exp(img/10, a_matrix, b_matrix, c_matrix) * 4
        img = img + color_bias + bias
        # img = img + color_bias
        return img.astype(np.float32)

    def add_color_bias_v4(self, img, parameter_a, parameter_b, parameter_c, band, bias_num):
        # bias = np.random.normal(scale=0.3)
        a_matrix = np.expand_dims(parameter_a, axis=1)
        a_matrix = np.repeat(a_matrix, img.shape[1], axis=1)
        b_matrix = np.expand_dims(parameter_b, axis=1)
        b_matrix = np.repeat(b_matrix, img.shape[1], axis=1)
        c_matrix = np.expand_dims(parameter_c, axis=1)
        c_matrix = np.repeat(c_matrix, img.shape[1], axis=1)
        # color_bias = logarithm(img/10, a_matrix, b_matrix) * 4
        color_bias = linear_exp(img/10, a_matrix, b_matrix, c_matrix) * 4

        bias = []
        for i in range(img.shape[2]):
            bias.append(np.random.choice(self.cb_bias_scale[band][i]) * bias_num)
        bias = np.array(bias)

        img = img + color_bias + bias
        # img = img + color_bias
        return img.astype(np.float32), (color_bias + bias).astype(np.float32)        


    def add_banding_noise(self, img, scale, RC=None):
        if RC == 'R':
            img = img + np.random.randn(1, img.shape[1], 1).astype(np.float32) * scale
        elif RC == 'C':
            img = img + np.random.randn(1, 1, img.shape[2]).astype(np.float32) * scale
        return img

class NoiseModelComplex:
    def __init__(self, mode=None):
        super(NoiseModelComplex, self).__init__()
        self.ratios = [0.1, 0.3, 0.5, 0.7]
        self.min_amount = 0.05
        self.max_amount = 0.15
        self.mode = mode
    
    def __call__(self, input, ratio=None):
        c,h,w = input.shape
        ### add gaussian noise
        sigma = 50
        output = input + sigma/255.0 * np.random.randn(c,h,w)

        ### add stripe noise
        b = random.sample(range(c), c)
        for k in b[:c//3]:
            [n] = random.sample(range(int(self.min_amount*h), int(self.max_amount*h), 1), 1)
            loc = random.sample(range(h), n)
            stripe = np.random.randn()*0.5-0.25
            for l in loc:
                output[k,l] -= stripe
        
        ### add deadline
        for k in b[c//3:2*c//3]:
            [n] = random.sample(range(int(self.min_amount*h), int(self.max_amount*h), 1), 1)
            loc = random.sample(range(h), n)
            for l in loc:
                output[k,l] = 0.0
        
        ### add impulse
        for k in b[2*c//3:]:
            [r] = random.sample(self.ratios, 1)
            mask = np.random.choice((0, 1, 2), size=(h, w), p=[1-r, r/2, r/2])
            img = output[k]
            img[mask==1] = 0
            img[mask==2] = 1
            output[k] = img
        
        return output.astype(np.float32)



##################################################################################################
class DataLoaderTrainNoise(Dataset):
    def __init__(self, rgb_dir, ratio=10, img_options=None, target_transform=None, start=0, end=45):
        super(DataLoaderTrainNoise, self).__init__()

        self.noise_model = NoiseModel()
        # self.noise_model = NoiseModel(model='g')
        # self.noise_model = NoiseModelComplex()

        # self.data_list = get_data_list()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input{}'.format(ratio))))
        
        self.clean_filenames = [os.path.join(rgb_dir, 'gt', x)          for x in clean_files[start:end] if is_tif_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, 'input{}'.format(ratio), x)     for x in noisy_files[start:end] if is_tif_file(x)]

        self.clean = [np.float32(load_tif_img(self.clean_filenames[index])) for index in range(len(self.clean_filenames))]
        self.noisy = [np.float32(load_tif_img(self.noisy_filenames[index])) for index in range(len(self.noisy_filenames))]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        self.ratio = ratio

    def __len__(self):
        return self.tar_size*6

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        # clean = np.float32(load_tif_img(self.clean_filenames[tar_index]))
        clean = self.clean[tar_index]
        noisy_real = self.noisy[tar_index]
        
        # clean = clean.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy_real = noisy_real[:, r:r + ps, c:c + ps]
        # h = clean.shape[1]//16*16
        # w = clean.shape[2]//16*16
        # clean = clean[:, :h, :w]
        noisy = self.noise_model(clean, ratio=self.ratio)
        noisy = noisy.clip(0, 1)
        # noisy = noisy[:, r:r + ps, c:c + ps]

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy) * self.ratio
        noisy_real = torch.from_numpy(noisy_real) * self.ratio
        # print(type(clean), type(noisy))

        apply_trans = transforms_aug[random.getrandbits(3)]

        # clean = getattr(augment, apply_trans)(clean)
        # noisy = getattr(augment, apply_trans)(noisy)        

        return noisy_real, noisy, clean_filename, clean_filename

##################################################################################################
class DataLoaderTrain(Dataset):
    # def __init__(self, rgb_dir, ratio=10, img_options=None, target_transform=None, start=0, end=45):
    def __init__(self, rgb_dir, list_path, ratio=10, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform

        self.data_list = get_data_list(list_path)
        # ratio_table = {10:'80', 20:'40'}
        ratio_table = {10:'80', 20:'40', 40:'20'}
        self.ratio_table = ratio_table

        scenes = [os.path.join(rgb_dir, d) for d in self.data_list]
        self.clean_filenames = [os.path.join(s, 'gt.npy') for s in scenes]
        # self.noisy_filenames = [os.path.join(s, ratio_table[ratio]+'.npy') for s in scenes]

        if ratio == 'mix':
            self.noisy_filenames = {}
            self.noisy = {}
            for r in ratio_table:
                self.noisy_filenames[r] = [os.path.join(s, ratio_table[r]+'.npy') for s in scenes]
                self.noisy[r] = [torch.from_numpy(np.float32(np.load(self.noisy_filenames[r][index]))) for index in range(len(self.noisy_filenames[r]))]
        else:
            self.noisy_filenames = [os.path.join(s, ratio_table[ratio]+'.npy') for s in scenes]
            self.noisy = [torch.from_numpy(np.float32(np.load(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]

        # clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))
        # noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'x{}'.format(ratio))))
        
        # self.clean_filenames = [os.path.join(rgb_dir, 'gt', x) for x in clean_files]
        # self.noisy_filenames = [os.path.join(rgb_dir, 'x{}'.format(ratio), x) for x in noisy_files]

        # self.clean = [torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        # self.noisy = [torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]

        self.clean = [torch.from_numpy(np.float32(np.load(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        self.ratio = ratio

    def __len__(self):
        return self.tar_size*6

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        # clean = torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[tar_index])))

        if self.ratio == 'mix':
            ratio = np.random.choice(list(list(self.ratio_table.keys())))
            noisy = self.noisy[ratio][tar_index]
            noisy_filename = os.path.split(self.noisy_filenames[ratio][tar_index])[-1]
        else:
            ratio = self.ratio
            noisy = self.noisy[tar_index]
            noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]


        clean = self.clean[tar_index]      

        # clean = clean.permute(2,0,1)
        # noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * ratio

        noisy = np.clip(noisy, 0, 1)

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        

        return clean, noisy, clean_filename, noisy_filename

##################################################################################################
class DataLoaderTrainSyn(Dataset):
    # def __init__(self, rgb_dir, ratio=10, img_options=None, target_transform=None, start=0, end=45):
    def __init__(self, rgb_dir, list_path, ratio=10, noisemodel='PGRC', img_options=None, target_transform=None):
        super(DataLoaderTrainSyn, self).__init__()

        self.target_transform = target_transform

        self.data_list = get_data_list(list_path)
        # ratio_table = {10:'80', 20:'40', 40:'20'}
        self.ratio_table = [10, 20, 40]

        scenes = [os.path.join(rgb_dir, d) for d in self.data_list]
        self.clean_filenames = [os.path.join(s, 'gt.npy') for s in scenes]
        # self.noisy_filenames = [os.path.join(s, ratio_table[ratio]+'.npy') for s in scenes]

        # clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))
        
        # self.clean_filenames = [os.path.join(rgb_dir, 'gt', x) for x in clean_files]

        self.clean = [np.float32(np.load(self.clean_filenames[index])) for index in range(len(self.clean_filenames))]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        self.ratio = ratio
        
        self.noise_type = noisemodel

        if noisemodel == 'complex':
            self.NoiseModel = NoiseModelComplex()
        elif '_' in noisemodel:
            self.NoiseModel = NoiseModel(noisemodel.split('_')[0], noisemodel.split('_')[1])
        else:
            self.NoiseModel = NoiseModel(noisemodel)

    def __len__(self):
        return self.tar_size*6

    def __getitem__(self, index):
        tar_index  = index % self.tar_size
        # clean = torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[tar_index])))
        clean = self.clean[tar_index]

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        # if clean_filename == 'scene_33.npy':
        #     print(clean_filename)

        if self.ratio == 'mix':
            ratio = np.random.uniform(low=10, high=40)
        else:
            ratio = self.ratio

        # if 'B' in self.noise_type:
        #     clean = np.clip(clean, 0+1e-6, 1)
        noisy = self.NoiseModel(clean, ratio)
        noisy = np.clip(noisy, 0, 1)

        # clean = clean.permute(2,0,1)
        # noisy = noisy.permute(2,0,1)


        # noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        

        return clean, noisy


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, list_path, ratio=10, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        self.data_list = get_data_list(list_path)
        # ratio_table = {10:'80', 20:'40'}
        ratio_table = {10:'80', 20:'40', 40:'20'}

        scenes = [os.path.join(rgb_dir, d) for d in self.data_list]
        self.clean_filenames = [os.path.join(s, 'gt.npy') for s in scenes]

        if ratio == 'mix':
            self.noisy_filenames = {}
            self.noisy = {}
            for r in ratio_table:
                self.noisy_filenames[r] = [os.path.join(s, ratio_table[r]+'.npy') for s in scenes]
                self.noisy[r] = [torch.from_numpy(np.float32(np.load(self.noisy_filenames[r][index]))) for index in range(len(self.noisy_filenames[r]))]
        else:
            self.noisy_filenames = [os.path.join(s, ratio_table[ratio]+'.npy') for s in scenes]
            self.noisy = [torch.from_numpy(np.float32(np.load(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]

        # clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))
        # # print(clean_files)
        # noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'x{}'.format(ratio))))


        # self.clean_filenames = [os.path.join(rgb_dir, 'gt', x)      for x in clean_files]
        # self.noisy_filenames = [os.path.join(rgb_dir, 'x{}'.format(ratio), x) for x in noisy_files]

        self.clean = [torch.from_numpy(np.float32(np.load(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        # self.noisy = [torch.from_numpy(np.float32(np.load(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]
        

        self.tar_size = len(self.clean_filenames)  
        self.ratio = ratio

    def __len__(self):
        if self.ratio == 'mix':
            return self.tar_size * 3
        else:
            return self.tar_size

    def __getitem__(self, index):
        tar_index  = index % self.tar_size
        # import pdb; pdb.set_trace()
        if self.ratio == 'mix':
            tar_ratio = index // self.tar_size
            if tar_ratio == 0:
                tar_r = 10
            elif tar_ratio == 1:
                tar_r = 20
            else:
                tar_r = 40
            
            noisy = self.noisy[tar_r][tar_index]
            noisy_filename = os.path.split(self.noisy_filenames[tar_r][tar_index])[-1]
        
        else:

            tar_r = self.ratio
            noisy = self.noisy[tar_index]
            noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        # clean = torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[tar_index])))
        clean = self.clean[tar_index]

                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]


        # clean = clean.permute(2,0,1)
        # noisy = noisy.permute(2,0,1)

        H = clean.shape[1]
        W = clean.shape[2]

        val_or_not = False
        if val_or_not:
            ps = 256
            r = clean.shape[1]//2-ps//2
            c = clean.shape[2]//2-ps//2
            clean = clean[:, r:r + ps, c:c + ps]
            noisy = noisy[:, r:r + ps, c:c + ps] * tar_r
        else:
            h = clean.shape[1]//16*16
            w = clean.shape[2]//16*16
            clean = clean[:, -h:, :w]
            noisy = noisy[:, -h:, :w] * tar_r
        
        noisy = np.clip(noisy, 0, 1)

        return clean, noisy, clean_filename, noisy_filename, H-h, 0


class DataLoaderVal_v2(Dataset):
    def __init__(self, rgb_dir, list_path, ratio=10, target_transform=None):
        super(DataLoaderVal_v2, self).__init__()

        self.target_transform = target_transform

        self.data_list = get_data_list(list_path)
        # ratio_table = {10:'80', 20:'40'}
        ratio_table = {10:'80', 20:'40', 40:'20'}

        scenes = [os.path.join(rgb_dir, d) for d in self.data_list]
        self.clean_filenames = [os.path.join(s, 'gt.npy') for s in scenes]

        if ratio == 'mix':
            self.noisy_filenames = {}
            self.noisy = {}
            for r in ratio_table:
                self.noisy_filenames[r] = [os.path.join(s, ratio_table[r]+'.npy') for s in scenes]
                self.noisy[r] = [torch.from_numpy(np.float32(np.load(self.noisy_filenames[r][index]))) for index in range(len(self.noisy_filenames[r]))]
        else:
            self.noisy_filenames = [os.path.join(s, ratio_table[ratio]+'.npy') for s in scenes]
            self.noisy = [torch.from_numpy(np.float32(np.load(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]

        # clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))
        # # print(clean_files)
        # noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'x{}'.format(ratio))))


        # self.clean_filenames = [os.path.join(rgb_dir, 'gt', x)      for x in clean_files]
        # self.noisy_filenames = [os.path.join(rgb_dir, 'x{}'.format(ratio), x) for x in noisy_files]

        self.clean = [torch.from_numpy(np.float32(np.load(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        # self.noisy = [torch.from_numpy(np.float32(np.load(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]
        
        self.row_loc = [0, 217-128]
        self.col_loc = [0, 409-128, (409-128)//2]

        self.tar_size = len(self.clean_filenames)  

        self.tar_size = self.tar_size * 6
        self.ratio = ratio

    def __len__(self):
        if self.ratio == 'mix':
            return self.tar_size * 3
        else:
            return self.tar_size

    def __getitem__(self, index):
        tar_index  = index % self.tar_size

        if self.ratio == 'mix':
            tar_ratio = index // self.tar_size
            if tar_ratio == 0:
                tar_r = 10
            elif tar_ratio == 1:
                tar_r = 20
            else:
                tar_r = 40
            
            noisy = self.noisy[tar_r][tar_index//6]
            noisy_filename = os.path.split(self.noisy_filenames[tar_r][tar_index//6])[-1]
        
        else:

            tar_r = self.ratio
            noisy = self.noisy[tar_index//6]
            noisy_filename = os.path.split(self.noisy_filenames[tar_index//6])[-1]


        # clean = torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[tar_index])))
        clean = self.clean[tar_index//6]

        patch_index = tar_index % 6
        row_index = patch_index // 3
        col_index = patch_index % 3
        row = self.row_loc[row_index]
        col = self.col_loc[col_index]
                
        clean_filename = os.path.split(self.clean_filenames[tar_index//6])[-1]


        # clean = clean.permute(2,0,1)
        # noisy = noisy.permute(2,0,1)

        clean = clean[:, row:row+128, col:col+128]
        noisy = noisy[:, row:row+128, col:col+128]

        val_or_not = False
        if val_or_not:
            ps = 256
            r = clean.shape[1]//2-ps//2
            c = clean.shape[2]//2-ps//2
            clean = clean[:, r:r + ps, c:c + ps]
            noisy = noisy[:, r:r + ps, c:c + ps] * tar_r
        else:
            h = clean.shape[1]//16*16
            w = clean.shape[2]//16*16
            clean = clean[:, :h, :w]
            noisy = noisy[:, :h, :w] * tar_r
        
        noisy = np.clip(noisy, 0, 1)

        return clean, noisy, clean_filename, noisy_filename

##################################################################################################

# class DataLoaderTest(Dataset):
#     def __init__(self, rgb_dir, list_path, target_transform=None):
#         super(DataLoaderTest, self).__init__()

#         self.target_transform = target_transform

#         noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))


#         self.noisy_filenames = [os.path.join(rgb_dir, 'input', x) for x in noisy_files if is_png_file(x)]
        

#         self.tar_size = len(self.noisy_filenames)  

#     def __len__(self):
#         return self.tar_size

#     def __getitem__(self, index):
#         tar_index   = index % self.tar_size
        

#         noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
#         noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

#         noisy = noisy.permute(2,0,1)

#         return noisy, noisy_filename
