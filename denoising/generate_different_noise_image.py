from cmath import nan
import numpy as np
import os
from os.path import join
from PIL import Image
import copy
import torch.nn.functional as F
import random
import scipy.stats as stats
import torch

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

        # import pdb; pdb.set_trace()
        self.cb_param = {}
        for band in range(25):
            self.cb_param[band] = np.load(os.path.join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/camera_params', 'band_{}_params.npy'.format(band)), allow_pickle=True).item()
            # print(self.cb_param[band]['color_bias'])

        # self.cb_bias_scale = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_exp/scale_mean.npy')
        self.cb_bias_scale = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_exp/cb_gau_scale_v2.npy', allow_pickle=True).item()

        # import pdb; pdb.set_trace()



    def _sample_params(self, band):
        Q_step = 1

        saturation_level = 1023
        profiles = ['Profile-1']

        camera_params = self.camera_params[band]
        camera_K = self.camera_K[band]

        # import pdb; pdb.set_trace()
        # if 'color_bias' in camera_params.keys():
        #     color_bias = np.random.choice(camera_params['color_bias'],1)
        if True: #TODO:
            color_bias = np.random.choice(self.cb_param[band]['color_bias'])
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
        output_cb = []

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
                cb = np.zeros_like(z)
                # z = self.add_color_bias_v0(z, color_bias, self.cb_log[band], self.cb_quadra['ab'][band], self.cb_quadra['ac'][band])
            elif 'B' in self.model:
                # z = self.add_color_bias_v2(z, np.mean(y), self.cb_log[band], self.cb_quadra['ab'][band], self.cb_quadra['ac'][band])
                # z = self.add_color_bias_v3(z, self.color_bias_a[idx:idx+1], self.color_bias_b[idx:idx+1],self.color_bias_c[idx:idx+1], self.cb_bias_scale[band])
                z, cb = self.add_color_bias_v4(z, self.color_bias_a[idx:idx+1], self.color_bias_b[idx:idx+1],self.color_bias_c[idx:idx+1], band, bias_num)
            else:
                cb = np.zeros_like(z)

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

            # if self.model == 'B':
            #     z = y + cb

            z_cb = (y + cb) * ratio / saturation_level
            output_cb.append(z_cb)

            z = z * ratio
            z = z / saturation_level
            output.append(z)

        output = np.concatenate(output)
        output_cb = np.concatenate(output_cb)

        if 'D' in self.model:
            for m in self.defects_mask:
                k, i, j = m
                output[k, i, j] = 1

        # return np.concatenate(output)
        return output, output_cb

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

        a_matrix = a_matrix[:,:,:img.shape[2]]
        b_matrix = b_matrix[:,:,:img.shape[2]]
        c_matrix = c_matrix[:,:,:img.shape[2]]

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
        
        return output.astype(np.float32), None


class NoiseModelDANet(object):
    def __init__(self):
        from models.UNetG import UNetG, sample_generator

        self.generator = UNetG(25)
        self.generator.load_state_dict(torch.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/DANet_ckpt/model_state_130.pt', map_location='cpu')['G'])

        self.device = torch.device('cuda:0')
        self.generator.to(self.device)
        self.generator.eval()

        self.generate_func = sample_generator
        

    def __call__(self, img, ratio=None):
        """
        img: c x h x w
        """
        img = torch.tensor(img).unsqueeze(0).to(self.device)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            noisy = self.generate_func(self.generator, img)
        noisy = noisy.to('cpu').numpy()[0]
        return noisy, None



if __name__ == "__main__":

    with open('data/dataset_v4/testval.txt', 'r') as f:
        scene_list = f.readlines()
        scene_list = [s.replace('\n', '') for s in scene_list]
    # scene_list.append('none')
    # import pdb; pdb.set_trace()
    ratios = [40] #[10, 20, 40]
    save_dir = 'img_noisemodel_raw'
    hw = (208, 400)
    if hw is None:
        hw = (217, 409)
    for ratio in ratios:
        for idx, scene in enumerate(scene_list):

            if idx != 21:
                continue #TODO:
            
            # long-exposure clean image and short-exposure image
            img_paths = [
                'data/scene_data_v4/{}/{}.npy'.format(scene, int(800/ratio)),
                'data/scene_data_v4/{}/gt.npy'.format(scene),
            ]
            for p in img_paths:
                img = np.load(p)
                img = img[:, 0:hw[0], 0:hw[1]]

                # save image to npy
                if 'gt' in p:
                    img_type = 'gt'
                else:
                    img_type = 'real_data'
                    img = img * ratio
                
                save_path = os.path.join(save_dir, img_type, str(ratio), '{}.npy'.format(idx))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, img)
                print('save to {}'.format(save_path))

                # save clean band to image
                for b in [2]: #TODO: only save the second band # range(noisy.shape[0]):
                    band = img[b]
                    band_img = (np.clip(band, 0, 1)*255).astype(np.uint8)
                    band_img = Image.fromarray(band_img)
                    save_path = os.path.join(save_dir, img_type, str(ratio), str(idx), 'band_{}.png'.format(b))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    band_img.save(save_path)

                
            # noise_models = ['g', 'Pg', 'PgR', 'PgRC', 'PgRCU', 'PgRCBU', 'P', 'R', 'C', 'U', 'B']
            noise_models_dict = {
                # 'ELD': 'PgRCbU',
                # 'HSI-Real': 'PgC',
                # 'Homo': 'g',
                # 'Hetero': 'pg',
                'Ours': 'PgRCBU',
                # 'DANet': 'DANet',
                # 'Complex': 'complex'
            }
            noise_models = list(noise_models_dict.values())


            for nm in noise_models:
                print(img.shape)
                save_dir_tmp = '{}/{}/{}/{}'.format(save_dir, nm, ratio, idx)
                os.makedirs(save_dir_tmp, exist_ok=True)

                if nm == 'complex':
                    noisemodel = NoiseModelComplex()
                elif nm == 'DANet':
                    noisemodel = NoiseModelDANet()
                else:
                    noisemodel = NoiseModel(nm)
                
                noisy, _ = noisemodel(img, ratio)

        
                # save the noisy image to npy
                save_path = os.path.join(save_dir_tmp, '..', '{}_noisy.npy'.format(idx))
                np.save(save_path, noisy)
                print('save to {}'.format(save_path))


                for b in [2]: #TODO: only save the second band # range(noisy.shape[0]):
                    band_noisy = noisy[b]
                    band_noise = noisy[b] - img[b]
                    band_noise = (band_noise - band_noise.min())


                    band_noisy = (np.clip(band_noisy, 0, 1)*255).astype(np.uint8)
                    band_noisy_img = Image.fromarray(band_noisy)
                    save_path = os.path.join(save_dir_tmp, 'band_{}.png'.format(b))
                    band_noisy_img.save(save_path)

                    # band_noise_img = (np.clip(band_noise, 0, 1)*255).astype(np.uint8)
                    # band_noise_img = Image.fromarray(band_noise_img)
                    # save_path = os.path.join(save_dir_tmp, 'noise_{}.png'.format(b))
                    # band_noise_img.save(save_path)

                    if nm in ['P']:
                        scale = 3
                    elif nm in ['B']:
                        scale = 5
                    elif nm in ['R', 'C']:
                        scale = 15
                    elif nm in ['U']: 
                        scale = 50 
                    else:  
                        scale = 1.5
                    band_noise_img_scale = (np.clip(band_noise*scale, 0, 1)*255).astype(np.uint8)
                    band_noise_img_scale = Image.fromarray(band_noise_img_scale)
                    save_path = os.path.join(save_dir_tmp, 'noise_scale_{}.png'.format(b))
                    band_noise_img_scale.save(save_path)

                    # print('save to {}'.format(save_path))