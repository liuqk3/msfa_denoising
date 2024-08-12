# from matplotlib.lines import _LineStyle
from statistics import median
import numpy as np
import os
from os.path import join
from read import read_file
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
from PIL import Image
import warnings
import cv2

def gauss(x, A, mu, sigma):
    
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def quadra(x, a, b, c):

    return a*x*x + b*x + c


def linear(x, a, b):
    return a*x+b

def logarithm(x, a, b):
    return a * np.log(x) + b

def logarithm_v2(x, a, b, c):
    return a * np.log(x+b) + c


def linear_exp(x, a, b, c):
    return a * np.exp(b*x) + c

def linear_exp_v2(x, a, b):
    return a * np.exp(b*x)


def calculate_R2(y, y_fitted):
    
    residuals = y - y_fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared =  1 - (ss_res / ss_tot)

    return r_squared

def median_filter(img, w):
    #median filter with window size of 2*k+1
    img_filter = np.zeros_like(img)
    for k in range(img.shape[0]):
        for i in range(img.shape[1]):
            for j in range(img.shape[2]):
                img_filter[k,i,j] = np.mean(img[k, max(0, i-k):min(img.shape[1], i+k+1),max(0, j-k):min(img.shape[2], j+k+1)])

    return img_filter

def color_bias_inbf():
    # path = '/home/jiangyuqi/Desktop/HSI_data/biasframe_1k'
    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/biasframe_v2'
    bf_list = os.listdir(path)

    r_squared_q = {}
    a_list = {}
    b_list = {}
    c_list = {}
    q_r2_list = {}

    # fit quadratic color bias
    for bf in bf_list:
        # bf_path = join(path, bf, 'unprocessed', 'frame.raw')
        # bf_data = read_file(bf_path)[:, 10:207-10, 10:409-10]
        bf_data = np.load(join(path, bf))

        for k in range(25):

            if k not in r_squared_q.keys():
                r_squared_q[k] = []
                a_list[k] = []
                b_list[k] = []
                c_list[k] = []
                q_r2_list[k] = []

            y = bf_data[k].mean(axis=0)
            y = np.array(y).astype(np.float32)
            x = np.array(range(bf_data.shape[2])).astype(np.float32) / 40

            coeff, var_matrix = curve_fit(quadra, x, y, p0=[-1,5,0])
            a, b, c = coeff
            y_fitted_q = quadra(x, a, b, c)
            r_squared_q[k].append(calculate_R2(y, y_fitted_q))
            a_list[k].append(a)
            b_list[k].append(b)
            c_list[k].append(c)

            # col_fitted = y_fitted_q.reshape(1, 409)
            # img = bf_data[k] - col_fitted
            # col_mean = img.mean(axis=0, keepdims=True).flatten()
            # _, (scale, loc, r) = stats.probplot(col_mean)
            # q_r2_list[k].append(r**2)

    tmp = []
    for k in range(25):
        tmp.append(np.mean(r_squared_q[k]))
    print(np.mean(tmp))

    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias/quadra_fit'
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(join(path, 'a-b'))
        os.mkdir(join(path, 'a-c'))

    # fit relationship within the quadratic function

    # for k in range(25):
    #     print(np.mean(a_list[k]))
    #     print(np.mean(b_list[k]))
    #     print(np.mean(c_list[k]))

    intra_frame_cb = {}
    ab = {}
    ac = {}

    r_ab_list = []
    r_ac_list = []

    for k in range(25):

        print("band "+str(k)+":")

        coeff, _ = curve_fit(linear, a_list[k], b_list[k])
        slope, intercept = coeff
        b_fitted = linear(np.array(a_list[k]), slope, intercept)
        r_square_ab = calculate_R2(b_list[k], b_fitted)
        r_ab_list.append(r_square_ab)

        ab[k] = {'slope':slope, 'intercept':intercept}

        plt.figure()
        plt.scatter(a_list[k], b_list[k])
        plt.plot(a_list[k], b_fitted)
        plt.savefig(join(path, 'a-b', 'band_'+str(k)+'.png'))
        plt.clf()
        print("R2 for a-b fitting:"+str(r_square_ab))

        coeff, _ = curve_fit(linear, a_list[k], c_list[k])
        slope, intercept = coeff
        c_fitted = linear(np.array(a_list[k]), slope, intercept)
        r_square_ac = calculate_R2(c_list[k], c_fitted)
        r_ac_list.append(r_square_ac)

        ac[k] = {'slope':slope, 'intercept':intercept}

        plt.figure()
        plt.scatter(a_list[k], c_list[k])
        plt.plot(a_list[k], c_fitted)
        plt.savefig(join(path, 'a-c', 'band_'+str(k)+'.png'))
        plt.clf()      
        print("R2 for a-c fitting:"+str(r_square_ac)) 

    intra_frame_cb['ab'] = ab
    intra_frame_cb['ac'] = ac
    print(np.mean(r_ab_list))
    print(np.mean(r_ac_list))



def estimate_cb():
    path = '/home/jiangyuqi/Desktop/HSI_data/flatframe_v5'

    ratio_table = {'01':0.1, '002':0.2}

    # gt = []
    # for i in range(1,2):
    #     gt.append(read_file(join(path, 'ff_400_'+str(i), 'unprocessed', 'frame.raw')))

    # gt = np.mean(gt, axis=0)

    ratio_list = ['01', '1', '5', '25', '50', '75', '100', '150', '200']

    data = {}

    for r in ratio_list:
        n = []
        for i in range(1,6):
            noise = read_file(join(path, 'ff_'+r+'_'+str(i), 'unprocessed', 'frame.raw'))
            n.append(noise)
        n = np.mean(n,axis=0)
        print(np.mean(n))
        print(np.shape(n))

        data[r] = n

    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias/log_fit'
    if not os.path.exists(path):
        os.mkdir(path)

    cb = {}
    mean_value = {}

    for r in ratio_list:
        for i in range(1, 6):
            # noise = []
            # for i in range(1, 2):
            #     noise.append(read_file(join(path, 'ff_'+r+'_'+str(i), 'unprocessed', 'frame.raw')))   

            # noise = np.mean(noise, axis=0)

            if r in ratio_table.keys():
                ratio = ratio_table[r]
            else:
                ratio = float(r)

            noise = data[r]
            gt = data['200']

            # print(r)
            # print(np.mean(noise))
            # print(np.mean(gt))

            for k in range(25):
                if k not in cb.keys():
                    cb[k] = []
                    mean_value[k] = []
                cb[k].append(cal_cb(np.mean(gt[k]), np.mean(noise[k]), 200/ratio)/4)
                mean_value[k].append(np.mean(noise[k])/1023)

    parameter = {}

    for k in range(25):

        print('band '+str(k))
        coef, _ = curve_fit(logarithm, mean_value[k], cb[k], p0=[-1,0,1])
        a, b, c =  coef
        print(b*np.array(mean_value[k])+c)
        fitted_cb = logarithm(np.array(mean_value[k]), a, b, c)
        r2 = calculate_R2(cb[k], fitted_cb)
        print(r2)

        parameter[k] = {'scale':a, 'slope':b, 'intercept':c}

        plt.figure()
        plt.scatter(mean_value[k], cb[k])
        plt.plot(mean_value[k], fitted_cb)
        plt.savefig(join(path, 'band_'+str(k)+'.png'))

    np.save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias', 'logarithm.npy'),parameter)


def cal_cb(gt, noise, ratio):
    return (noise * ratio - gt)/(ratio)


def estimate_cb_v2():
    path = '/home/jiangyuqi/Desktop/HSI_data/flatframe_v5'

    ratio_table = {'01':0.1, '002':0.2}

    # gt = []
    # for i in range(1,2):
    #     gt.append(read_file(join(path, 'ff_400_'+str(i), 'unprocessed', 'frame.raw')))

    # gt = np.mean(gt, axis=0)

    ratio_list = ['01', '1', '5', '25', '50', '75', '100', '150', '200']

    data = {}

    for r in ratio_list:
        n = []
        for i in range(1,6):
            noise = read_file(join(path, 'ff_'+r+'_'+str(i), 'unprocessed', 'frame.raw'))
            n.append(noise)
        n = np.mean(n,axis=0)
        print(np.mean(n))
        print(np.shape(n))

        data[r] = n

    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v2/exp_fit'
    if not os.path.exists(path):
        os.makedirs(path)

    cb = {}
    mean_value = {}

    for r in ratio_list:

        if r == '200':
            continue

        for i in range(1, 6):
            # noise = []
            # for i in range(1, 2):
            #     noise.append(read_file(join(path, 'ff_'+r+'_'+str(i), 'unprocessed', 'frame.raw')))   

            # noise = np.mean(noise, axis=0)

            if r in ratio_table.keys():
                ratio = ratio_table[r]
            else:
                ratio = float(r)

            noise = data[r]
            gt = data['200']

            # print(r)
            # print(np.mean(noise))
            # print(np.mean(gt))

            for k in range(25):
                if k not in cb.keys():
                    cb[k] = []
                    mean_value[k] = []
                cb[k].append(cal_cb(np.mean(gt[k]), np.mean(noise[k]), 200/ratio)/4)
                mean_value[k].append(np.mean(noise[k])/1023)

    parameter = {}

    r2 = []
    for k in range(25):

        print('band '+str(k))
        # coef, _ = curve_fit(linear_exp, mean_value[k], cb[k], p0=[1,-20,0])
        coef, _ = curve_fit(logarithm, mean_value[k], cb[k], )
        a, b =  coef
        # print(b*np.array(mean_value[k])+c)
        fitted_cb = logarithm(np.array(mean_value[k]), a, b)
        r2.append(calculate_R2(cb[k], fitted_cb))
        # print(r2)

        parameter[k] = {'scale':a, 'slope':b, }

        fit_x = np.array(range(60))/100
        fit_y = logarithm(fit_x,a,b)

        plt.figure()
        plt.scatter(mean_value[k], cb[k])
        plt.plot(fit_x, fit_y)
        plt.savefig(join(path, 'band_'+str(k)+'.png'))
    print(np.mean(r2))
    # np.save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v2', 'logarithm.npy'),parameter)


def estimate_cb_v3():

    fig_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_test'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    path = '/home/jiangyuqi/Desktop/HSI_data/ff_cb_v10'

    ratio_table = {'01':0.1, '002':0.2}

    # gt = []
    # for i in range(1,2):
    #     gt.append(read_file(join(path, 'ff_400_'+str(i), 'unprocessed', 'frame.raw')))

    # gt = np.mean(gt, axis=0)

    ratio_list = ['20','30', '40','50','60', '80','900']
    # ratio_list = ['1', '5', '10', '20','30', '40','50','60', '80','900']
    # ratio_list = ['20','30','40','50','60','80','90','100','150','200','300','900']

    data = {}

    for r in ratio_list:
        # if r == '800':
        #     continue
        # n = []
        # if r == '60':
        #     num = 20
        # else:
        #     num = 40
        n = []
        num=10
        for i in range(num):
            n.append(read_file(join(path, 'ff_'+r+'_'+str(i+1), 'unprocessed', 'frame.raw'))) 
        n = np.mean(n,axis=0)

        n = median_filter(n, 3)
        data[r] = n

        img = (np.clip(data[r]*900/float(r), 0, 1023)/1023*255).astype(np.uint8)
        Image.fromarray(img[0]).save('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_test/'+str(r)+'.png')

        # data[r] = read_file(join(path, 'ff_'+r, 'unprocessed', 'frame.raw'))
        print(np.mean(data[r])*900/float(r))
        # print(np.mean(np.clip(data[r]*950/float(r), 0, 1023)))

    # data['800'] = read_file(join(path, 'ff_'+r+'_20', 'unprocessed', 'frame.raw'))

    for r in ratio_list:
        noise = data[r]
        gt = data['900']
        print(r)
        print(cal_cb(np.mean(gt), np.mean(noise), 900/float(r)))

    # parameter = {}
    a_matrix = np.zeros((25, 217, 409))
    b_matrix = np.zeros((25, 217, 409))
    c_matrix = np.zeros((25, 217, 409))
    r2_matrix = np.zeros((25, 217, 409))

    for k in range(25):
        for i in range(217):
            r2_list = []
            for j in range(409):

                # if (k,i,j) not in parameter.kesy():
                #     parameter

                # print((k,i,j))

                cb = []
                value = []

                for r in ratio_list:
                    if r == '900':
                        continue

                    if r in ratio_table.keys():
                        ratio = ratio_table[r]
                    else:
                        ratio = float(r)

                    noise = data[r]
                    gt = data['900']
                    np.clip(noise, 0, 1023)

                    cb.append(cal_cb(gt[k,i,j], noise[k,i,j], 900/ratio)/4)
                    value.append((noise[k,i,j]-cal_cb(gt[k,i,j], noise[k,i,j], 900/ratio))/100) 

                cb = np.array(cb)
                value = np.array(value)

                coeff, _ = curve_fit(linear_exp, value, cb, p0=[1,-1,0], maxfev=100000)
                a, b, c = coeff
                fitted_cb = linear_exp(value, a, b, c)
                
                # coeff, _ = curve_fit(linear_exp_v2)
                
                r2 = calculate_R2(cb, fitted_cb)
                # print(r2)
                r2_matrix[k, i, j] = r2
                a_matrix[k,i,j] = a
                b_matrix[k,i,j] = b
                c_matrix[k,i,j] = c

                r2_list.append(r2)

                plt.figure()
                plt.scatter(value, cb)
                plt.plot(value, fitted_cb)
                plt.savefig(join(fig_path, str(k)+'_'+str(i)+'_'+str(j)+'.png'))
                plt.clf()

            print(np.mean(r2_list))
    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/color_bias_v3'
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(join(path, 'r2_matrix.npy'),r2_matrix)
    np.save(join(path, 'a_matrix.npy'),a_matrix)
    np.save(join(path, 'b_matrix.npy'),b_matrix)
    np.save(join(path, 'c_matrix.npy'),c_matrix)
    print(np.mean(r2_matrix))



def estimate_cb_v4():

    fig_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_test_v100'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    path = '/home/jiangyuqi/Desktop/HSI_data/ff_cb_v10'

    ratio_table = {'01':0.1, '002':0.2}

    ratio_list = ['1', '5', '10', '20','30', '40','50','60', '80','900']

    data = {}

    for r in ratio_list:

        n = []
        num=10
        for i in range(num):
            n.append(read_file(join(path, 'ff_'+r+'_'+str(i+1), 'unprocessed', 'frame.raw'))) 
        n = np.mean(n,axis=0)

        data[r] = n


    for r in ratio_list:
        noise = data[r]
        gt = data['900']
        if not os.path.exists('flatfield_frame'):
            os.mkdir('flatfield_frame')
        cv2.imwrite(join('flatfield_frame', r+'.png'), data[r][0]*900/int(r))
        print(r)
        print(cal_cb(np.mean(gt), np.mean(noise), 900/float(r)))

    # parameter = {}
    a_matrix = np.zeros((25, 409))
    b_matrix = np.zeros((25, 409))
    c_matrix = np.zeros((25, 409))
    r2_matrix = np.zeros((25, 409))

    for k in range(25):
        r2_list = []
        for j in range(409):

            cb = []
            value = []

            for r in ratio_list:
                if r == '900':
                    continue

                if r in ratio_table.keys():
                    ratio = ratio_table[r]
                else:
                    ratio = float(r)

                noise = data[r]
                gt = data['900']
                # np.clip(noise, 0, 1023)
                gt_col = np.mean(gt[k], axis=0)[j]
                noise_col = np.mean(noise[k], axis=0)[j]

                cb.append(cal_cb(gt_col, noise_col, 900/ratio)/4)
                value.append((noise_col-cal_cb(gt_col, noise_col, 900/ratio))/10) 

            cb = np.array(cb)
            value = np.array(value)

            warnings.filterwarnings('ignore')
            coeff, _ = curve_fit(linear_exp, value, cb, p0=[1,-1,0], maxfev=5000)
            a, b, c = coeff
            fitted_cb = linear_exp(value, a, b, c)
            # coeff, _ = curve_fit(linear_exp_v2, value, cb, p0=[1,-1], maxfev=5000)
            # a,b = coeff
            # fitted_cb = linear_exp_v2(value, a, b)
            # coeff, _ = curve_fit(logarithm, value, cb, maxfev=10000)
            # a, b = coeff
            # fitted_cb = logarithm(value, a, b)
            # coeff, _ = curve_fit(logarithm_v2, value, cb, p0=[-1, 1, 0], maxfev=5000)
            # a, b, c = coeff
            # fitted_cb = logarithm_v2(value, a, b, c)
            r2 = calculate_R2(cb, fitted_cb)
            # print(r2)
            r2_matrix[k,j] = r2
            a_matrix[k,j] = a
            b_matrix[k,j] = b
            # c_matrix[k,j] = c

            r2_list.append(r2)

            plt.figure()
            plt.plot(value, cb,'bo', label='Data Point', markersize=7)
            plt.plot(value, fitted_cb, 'r-', label='Fitted Curve', linewidth=3.0)
            # plt.annotate('$R^2 = {:.3f}$'.format(r2), xy=(0.8, 0.3), xycoords='axes fraction', fontsize=18)
            plt.xlabel('Light intensity', fontsize=18)
            plt.ylabel(r'$\mu_{b}^c$', fontsize=18)
            plt.title('Fitting Curve'+' ($R^2 = {:.3f}$)'.format(r2), fontsize=18)
            # plt.legend()
            plt.savefig(join(fig_path, str(k)+'_'+str(j)+'.pdf'), bbox_inches='tight')
            plt.clf()
        print(k)
        print(np.mean(r2_list))
    # path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_log'
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # np.save(join(path, 'r2_matrix.npy'),r2_matrix)
    # np.save(join(path, 'a_matrix.npy'),a_matrix)
    # np.save(join(path, 'b_matrix.npy'),b_matrix)
    # np.save(join(path, 'c_matrix.npy'),c_matrix)
    # print(np.mean(r2_matrix))


def estimate_cb_v7():

    fig_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_test_v100'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    path = '/home/jiangyuqi/Desktop/HSI_data/ff_cb_v10'

    ratio_table = {'01':0.1, '002':0.2}

    ratio_list = ['1', '5', '10', '20','30', '40','50','60', '80','900']

    data = {}

    for r in ratio_list:

        n = []
        num=10
        for i in range(num):
            n.append(read_file(join(path, 'ff_'+r+'_'+str(i+1), 'unprocessed', 'frame.raw'))) 
        n = np.mean(n,axis=0)

        data[r] = n


    for r in ratio_list:
        noise = data[r]
        gt = data['900']
        if not os.path.exists('flatfield_frame'):
            os.mkdir('flatfield_frame')
        cv2.imwrite(join('flatfield_frame', r+'.png'), data[r][0]*900/int(r))
        # print(r)
        # print(cal_cb(np.mean(gt), np.mean(noise), 900/float(r)))

    # parameter = {}
    a_matrix = np.zeros((25, 217, 409))
    b_matrix = np.zeros((25, 217, 409))
    c_matrix = np.zeros((25, 217, 409))
    r2_matrix = np.zeros((25, 217, 409))

    r2_list = []

    for k in range(25):
        for i in range(217):

            for j in range(409):

                cb = []
                value = []

                for r in ratio_list:
                    if r == '900':
                        continue

                    if r in ratio_table.keys():
                        ratio = ratio_table[r]
                    else:
                        ratio = float(r)

                    noise = data[r]
                    gt = data['900']
                    # np.clip(noise, 0, 1023)
                    # gt_col = np.mean(gt[k], axis=0)[j]
                    gt_value = gt[k,i,j]
                    noise_value = noise[k,i,j]
                    # noise_col = np.mean(noise[k], axis=0)[j]

                    cb.append(cal_cb(gt_value, noise_value, 900/ratio)/4)
                    value.append((noise_value-cal_cb(gt_value, noise_value, 900/ratio))/10) 

                cb = np.array(cb)
                value = np.array(value)

                warnings.filterwarnings('ignore')
                coeff, _ = curve_fit(linear_exp, value, cb, p0=[1,-1,0], maxfev=5000)
                a, b, c = coeff
                fitted_cb = linear_exp(value, a, b, c)

                r2 = calculate_R2(cb, fitted_cb)
                # print(r2)
                r2_matrix[k,i,j] = r2
                a_matrix[k,i,j] = a
                b_matrix[k,i,j] = b
                c_matrix[k,i,j] = c
                # print(r2)

                r2_list.append(r2)

                plt.figure()
                plt.plot(value, cb,'bo', label='Data Point', markersize=7)
                plt.plot(value, fitted_cb, 'r-', label='Fitted Curve', linewidth=3.0)
                # plt.annotate('$R^2 = {:.3f}$'.format(r2), xy=(0.8, 0.3), xycoords='axes fraction', fontsize=18)
                plt.xlabel('Light intensity', fontsize=18)
                plt.ylabel(r'$\mu_{b}^c$', fontsize=18)
                plt.title('Fitting Curve'+' ($R^2 = {:.3f}$)'.format(r2), fontsize=18)
                # plt.legend()
                if i < 10:
                    plt.savefig(join(fig_path, str(k)+'_'+str(i)+'_'+str(j)+'.png'), bbox_inches='tight')
                plt.clf()
        print(k)
        print(np.mean(r2_list))




def bf_color_bias_gaussian():

    cb_path ='/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_exp'
    a_matrix = np.load(join(cb_path, 'a_matrix.npy'))
    b_matrix = np.load(join(cb_path, 'b_matrix.npy'))
    c_matrix = np.load(join(cb_path, 'c_matrix.npy'))

    x = (a_matrix+c_matrix) * 4

    bf_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/biasframe_v2'    
    index = []
    scale_val = []
    for k in range(25):
        bias = []
        for i in range(20):
            data = np.load(join(bf_path, 'bf_'+str(i+1)+'.npy'))
            # print(np.mean(data))
            bias.append(np.mean(data[k])-np.mean(x[k]))

        # if k == 24:
        _, scale = stats.norm.fit(np.array(bias), floc=0)
        scale_val.append(scale)

    np.save(join(cb_path, 'scale_mean.npy'), np.array(scale_val))

def refine_plot(ax, r=None, fontsize=18):
    # plt.xlabel('Theoretical quantiles', fontsize=18)
    # plt.ylabel('Ordered values', fontsize=18)
    # plt.title('Normal Probability Plot', fontsize=18)
    
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)


    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    if r is not None:
        # plt.annotate('$R^2 = {:.3f}$'.format(r**2), xy=(0.6, 0.1), xycoords='axes fraction', fontsize=fontsize)
        plt.title('Probability Plot ($R^2 = {:.3f}$)'.format(r**2), fontsize=fontsize)
    ax.title.set_fontsize(fontsize)

def fit_colorbias():
    bf_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/biasframe_v2'    
    color_bias_mean = []
    a_matrix = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/color_bias_v3_exp')
    for k in range(25):
        for i in range(20):
            data = np.load(join(bf_path, 'bf_'+str(i+1)+'.npy'))
            print(np.mean(data[k]))
            color_bias_mean.append(np.mean(data[k]))

        x = np.array(color_bias_mean)
        ax = plt.subplot(111)
        _, (scale, loc, r) = stats.probplot(x, plot=ax)
        print(loc)
        print(scale)
        refine_plot(ax, r)
        # plt.show()
        if not os.path.exists('figure_withcb/color_bias'):
            os.mkdir('figure_withcb/color_bias')
        plt.savefig('figure_withcb/color_bias/band_{}_color_bias.png'.format(k), bbox_inches='tight')
        plt.clf()

def ff_cb_bias():
    path = '/home/jiangyuqi/Desktop/HSI_data/ff_cb_v10'

    ratio_table = {'01':0.1, '002':0.2}

    ratio_list = ['1', '5', '10', '20','30', '40','50','60', '80','900']


    for r in ratio_list:

        if r == '900':
            continue

        n = []
        num=10
        for i in range(num):
            d = read_file(join(path, 'ff_'+r+'_'+str(i+1), 'unprocessed', 'frame.raw'))
            n.append(np.mean(d))

        x = np.array(n)
        ax = plt.subplot(111)
        _, (scale, loc, r2) = stats.probplot(x, plot=ax)
        print(loc)
        print(scale)
        refine_plot(ax, r2)
        # plt.show()
        # plt.title()
        if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/color_bias_test_v3'):
            os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/color_bias_test_v3')
        plt.savefig('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/color_bias_test_v3/'+str(r)+'.png', bbox_inches='tight')
        plt.clf()




if __name__ == '__main__':

    estimate_cb_v7()
