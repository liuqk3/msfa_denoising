import numpy as np
import os
from os.path import join, splitext
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from scipy.linalg import lstsq
import cv2
import scipy.stats as stats
from scipy.io import savemat, loadmat
from skimage import io
import scipy.stats as stats
import seaborn as sns
sns.set(color_codes=True)
from collections import namedtuple
import glob
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def quadra(x, a, b, c):

    return a*x*x + b*x + c

def remove_color_bias(img):
    y = np.mean(img, axis=0)
    x = np.array(range(img.shape[1])).astype(np.float32) / 40
    coeff, var_matrix = curve_fit(quadra, x, y)
    a, b, c = coeff
    y_fitted_q = quadra(x, a, b, c)

    img = img - y_fitted_q.reshape(1,-1)
    return img

def remove_cb_v2(img,k):
    a_matrix = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_exp/a_matrix.npy')
    b_matrix = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_exp/b_matrix.npy')
    c_matrix = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_exp/c_matrix.npy')

    cb_col = (a_matrix + c_matrix)*4
    cb_col = np.expand_dims(cb_col, axis=1)
    cb_col = cb_col[k,:, 10:409-10]
    cb_col = cb_col - (np.mean(cb_col)-np.mean(img))
    img = img - cb_col
    return img

class NoiseEstimator:
    def infer_parameters(self, img):  # C H W



        np.random.seed()
        param_dic = {}

        # img = remove_color_bias(img)
        # img = remove_cb_v2(img)

        #color bias
        # param_dic['color_bias'] = np.mean(img)
        # img = img - np.mean(img)
        # print(param_dic['color_bias'])

        scale, img = self.estimate_col_noise(img)
        param_dic['C'] = {'scale': scale}
        print(np.amin(img), scale)
        scale, img = self.estimate_row_noise(img)
        param_dic['R'] = {'scale': scale}
        print(np.amin(img), scale)
        # scale, img = self.estimate_col_noise(img)
        # param_dic['C'] = {'scale': scale}
        # print(np.amin(img), scale)
        np.random.seed(0)

        # random sampling?
        img_flip = np.concatenate((img[img > 0], -1.0 * img[img > 0]))
        # print(img_flip.flatten().shape)
        x = np.random.choice(img_flip.flatten(), size=min(20000, img_flip.flatten().shape[0]), replace=False)
        
        scale, r = self.fit_norm_dist(x)
        param_dic['G'] = {'scale': scale, 'R2': r**2}
        scale, shape, r = self.fit_tukeylambda_dist(x)
        param_dic['TL'] = {'scale': scale, 'shape': shape, 'R2': r**2}
        print(param_dic)
        return param_dic

    def estimate_row_noise(self, img):
        x = img.mean(axis=1, keepdims=True)
        print(x.shape, img.shape)
        _, scale = stats.norm.fit(x)
        img = img - x
        return scale, img
    
    def estimate_col_noise(self, img):
        x = img.mean(axis=0, keepdims=True)
        print(x.shape, img.shape)
        _, scale = stats.norm.fit(x)
        img = img - x
        return scale, img

    def fit_tukeylambda_dist(self, x):
        svals, ppcc = stats.ppcc_plot(x, -0.5, 0.2, N=50)
        best_shape_val = svals[np.argmax(ppcc)]
        _, (scale, _, r) = stats.probplot(
            x, best_shape_val, dist='tukeylambda')
        return scale, best_shape_val, r

    def fit_norm_dist(self, x):
        _, (scale, _, r) = stats.probplot(x)
        return scale, r

def estimate_noise_params(k, bf_data):
    estimator = NoiseEstimator()
    print('init param_dict...')
    data = []
    print('process bias frame...')

    for i in range(20):
        bf = bf_data[i][k]
        # bf = bf[10:207, 10:399]
        # bf = bf - color_bias
        bf = remove_cb_v2(bf, k)
        param_dic = estimator.infer_parameters(bf)
        G_scale = param_dic['G']['scale']
        TL_shape = param_dic['TL']['shape']
        TL_scale = param_dic['TL']['scale']
        R_scale = param_dic['R']['scale']
        C_scale = param_dic['C']['scale']
        # color_bias = param_dic['color_bias']
        data.append((k, G_scale, TL_shape, TL_scale, R_scale, C_scale))
        # data.append((k, G_scale, TL_shape, TL_scale, R_scale, C_scale, color_bias))

    return data


def joint_params_dist(k, data):
    # iso = 800
    K = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/K/color_{}.npy'.format(k), allow_pickle=True).item()['K']
    print(K)


    camera_params = {}
    # camera_params['Kmin'] = K / 8
    # camera_params['Kmax'] = K * 8

    D = [tuple(data[i]) for i in range(len(data))]
    D = list(zip(*D))

    X = np.array(D[0])

    X = K
    # X = (X / iso * K)  # all the system gains K
    X = np.log(X)  # log_K

    labels = ['k', 'g_scale', 'G_shape', 'G_scale', 'R_scale', 'C_scale']
    camera_params['Profile-1'] = {}

    for i in [2]:
        y = np.array(D[i])   # TL shape 
        camera_params[labels[i]] = y

    # camera_params['Profile-1']['color_bias'] = D[6]

    for i in [1, 3, 4, 5]:  # log scale for TL scale, row noise scale, colum noise scale
        y = np.array(D[i])
        x = X
        y = y
        y = np.log(y)

        # slope, intercept, r, prob, sterrest = stats.linregress(x, y)
        slope = 0.0
        intercept = np.mean(y)
        y_est = x * slope + intercept
        rss = np.sum((y - y_est)**2)

        err_std = np.sqrt(rss / (len(y) - 2))

        print(k, i, slope, intercept, err_std)

        camera_params['Profile-1'][labels[i]] = {}
        camera_params['Profile-1'][labels[i]]['slope'] = slope
        camera_params['Profile-1'][labels[i]]['bias'] = intercept
        camera_params['Profile-1'][labels[i]]['sigma'] = err_std

        # ax = plt.subplot(111)
        # sns.regplot(x, y, 'ci', ax=ax)
        # plt.plot(x, y, 'bo', x, y_est, 'r-')
        # plt.fill_between(x, y_est-1.96*err_std, y_est+1.96*err_std, alpha=0.5, linewidth=3, color='b', interpolate=True)
        # plt.xlabel('$\log(K)$', fontsize=18)

        # if i == 3:
        #     plt.ylabel('$\log(\sigma_{TL})$', fontsize=18)
        # elif i == 4:
        #     plt.ylabel('$\log(\sigma_{r})$', fontsize=18)
        # elif i == 5:
        #     plt.ylabel('$\log(\sigma_{c})$', fontsize=18)
        # # plt.show()

        # plt.savefig('figure/joint_distribution/band_{}_{}.pdf'.format(k, i), bbox_inches='tight')
        # plt.clf()

    # camera_params['Profile-1']['color_bias'] = color_bias

    return camera_params



if __name__ == '__main__':
    # cameras = ['x1', 'wide']
    # suffixes = ['.raw', 'raw']

    # for camera, suffix in zip(cameras, suffixes):
    #     print('----------- {} -----------'.format(camera))

    #     # param analysis
    #     basedir = 'dataset'
    #     rawpaths = sorted(glob.glob(join(basedir, 'BiasFrames', camera, '*{}'.format(suffix))))
    #     data_path = join('camera_params', 'P40ProPlus_'+camera+'.npy')
    #     if os.path.exists(data_path):
    #         data = np.load(data_path, allow_pickle=True)
    #     else:
    #         data = estimate_noise_params(rawpaths, camera)

    #     if not os.path.exists(data_path):
    #         np.save(data_path, data)

    #     camera_params = joint_params_dist(basedir, data)
    #     params_path = join('camera_params', 'P40ProPlus_{}.npy'.format(camera+'_params'))

    #     if not os.path.exists(params_path):
    #         print('save {}..'.format(params_path))
    #         np.save(params_path, camera_params)
    # if not os.path.exists('camera_params_withcb'):
    #     os.mkdir('camera_params_withcb')
    # if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/figure_withcb/joint_distribution'):
    #     os.makedirs('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/figure_withcb/joint_distribution')


    # for k in range(25):

    #     data_path = join('camera_params_withcb', 'band_{}.npy'.format(k))
    #     # if os.path.exists(data_path):
    #     #     data = np.load(data_path, allow_pickle=True)
    #     if False:
    #         pass
    #     else:
    #         bf_data = []
    #         cb = []
    #         for i in range(20):
    #             bf_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/biasframe_v2'
    #             bf = np.load(join(bf_path, 'bf_'+str(i+1)+'.npy'))
    #             # bf = bf[:, 109-70:109+70, 205-70:205+70]
    #             bf = bf[:, 10:207-10, 10:409-10]
    #             # cb.append(np.mean(bf[k]))
    #             # for l in range(25):
    #             #     bf[l] = bf[l] - np.mean(bf[l])

    #             bf_data.append(bf)

    #         # color_bias = np.mean(cb)
    #         data = estimate_noise_params(k, bf_data)
        
    #     if not os.path.exists(data_path):
    #         np.save(data_path, data)

    #     camera_params = joint_params_dist(k, data)
    #     params_path = join('camera_params_withcb', 'band_{}_params.npy'.format(k))

    #     # camera_params['color_bias'] = cb

    #     if not os.path.exists(params_path):
    #         print('save {}..'.format(params_path))
    #     np.save(params_path, camera_params)

    if not os.path.exists('bf_img'):
        os.mkdir('bf_img')
    # if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/figure_withcb/joint_distribution'):
    #     os.makedirs('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/figure_withcb/joint_distribution')


    for k in range(25):

        # data_path = join('camera_params_withcb', 'band_{}.npy'.format(k))
        # if os.path.exists(data_path):
        #     data = np.load(data_path, allow_pickle=True)
        if False:
            pass
        else:
            bf_data = []
            cb = []
            for i in range(20):
                bf_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/biasframe_v2'
                bf = np.load(join(bf_path, 'bf_'+str(i+1)+'.npy'))
                # bf = bf[:, 109-70:109+70, 205-70:205+70]
                bf = bf[:, 10:207-10, 10:409-10]
                # cb.append(np.mean(bf[k]))
                # for l in range(25):
                #     bf[l] = bf[l] - np.mean(bf[l])
                bf = remove_cb_v2(bf, k)
                from PIL import Image
                img = np.clip(bf[k]*100,0,1023)/4
                Image.fromarray(img.astype(np.uint8)).save('bf_img/bf.png')
                break
        break
                # bf_data.append(bf)

            # color_bias = np.mean(cb)
        #     data = estimate_noise_params(k, bf_data)
        
        # if not os.path.exists(data_path):
        #     np.save(data_path, data)

        # camera_params = joint_params_dist(k, data)
        # params_path = join('camera_params_withcb', 'band_{}_params.npy'.format(k))

        # # camera_params['color_bias'] = cb

        # if not os.path.exists(params_path):
        #     print('save {}..'.format(params_path))
        # np.save(params_path, camera_params)