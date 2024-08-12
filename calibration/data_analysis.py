import numpy as np
import os
from os.path import join, splitext
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import scipy.stats as stats
from collections import namedtuple
from skimage import io
import cv2
import glob
from scipy.optimize import curve_fit

def quadra(x, a, b, c):

    return a*x*x + b*x + c

def crop_center(img,cropx,cropy): # C * H * W
    _,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty:starty+cropy,startx:startx+cropx]


def refine_plot(ax, r=None, fontsize=18):
    # plt.xlabel('Theoretical quantiles', fontsize=18)
    # plt.ylabel('Ordered values', fontsize=18)
    # plt.title('Normal Probability Plot', fontsize=18)
    
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)
    ax.title.set_fontsize(fontsize)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    if r is not None:
        plt.annotate('$R^2 = {:.3f}$'.format(r**2), xy=(0.6, 0.1), xycoords='axes fraction', fontsize=fontsize)

def remove_cb(img):
    y = img.mean(axis=1)
    y = np.array(y).astype(np.float32)
    x = np.array(range(img.shape[0])).astype(np.float32) / 40
    coeff, var_matrix = curve_fit(quadra, x, y)
    a, b, c = coeff
    y_fitted_q = quadra(x, a, b, c)

    img = img - y_fitted_q.reshape((389,1))

    return img


def BF_ISO_analysis(k=128):
    # Image = io.imread('/media/zhangtao/UUI/HSI/color/2021-01-19-1556_I00.1_4.tif').astype(np.float32)
    bf = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/biasframe_v2/bf_1.npy')
    # bf = bf[:, 109-70:109+70, 205-70:205+70]
    bf = bf[:, 10:217-10, 10: 409-10]
    print('Process: {}'.format(k))
    
    img = bf[k]
    img = np.rot90(img)
    print(img.shape)
    # cv2.imshow('0', img/np.amax(img))
    # cv2.waitKey()
    # return

    #color bias
    # img = img - np.mean(img)
    img = remove_cb(img)

    banding_FFT(img, k)
    # analysis row noise
    x = img[...].mean(axis=1).flatten()
    ax = plt.subplot(111)
    _, (scale, loc, r) = stats.probplot(x, plot=ax)
    print(scale)
    refine_plot(ax, r)
    # plt.show()
    plt.savefig('figure/BF_analysis/band_{}_row.png'.format(k), bbox_inches='tight')
    plt.clf()
    x = img.mean(axis=1, keepdims=True)
    img = img - x

    # analysis col noise
    x = img[...].mean(axis=0).flatten()
    ax = plt.subplot(111)
    _, (scale, loc, r) = stats.probplot(x, plot=ax)
    print(scale)
    refine_plot(ax, r)
    # plt.show()
    plt.savefig('figure/BF_analysis/band_{}_col.png'.format(k), bbox_inches='tight')
    plt.clf()
    x = img.mean(axis=0, keepdims=True)
    img = img - x

    # x = np.random.choice(img[...].flatten(), size=50000, replace=False)
    # x = np.random.choice(img[img > 0].flatten(), size=100000, replace=False)
    img_flip = np.concatenate((img[img > 0], -1.0 * img[img > 0]))
    x = np.random.choice(img_flip.flatten(), size=5000, replace=False)

    ax = plt.subplot(111)
    _, (scale, loc, r) = stats.probplot(x, plot=ax)
    ax.set_title('Probability Plot')
    refine_plot(ax, r, fontsize=20)

    # plt.show()
    plt.savefig('figure/BF_analysis/band_{}_read_gauss.png'.format(k), bbox_inches='tight')        
    plt.clf()
    
    ax = plt.subplot(111)
    svals, ppcc = stats.ppcc_plot(x, -0.5, 0.2, plot=ax, N=50)
    best_shape_val = svals[np.argmax(ppcc)]

    plt.annotate('$\lambda = {:.2f}$'.format(best_shape_val), xy=(0.7, 0.1), xycoords='axes fraction', fontsize=20)
    ax.vlines(best_shape_val, 0, 1, colors='r', label='Expected shape value')
    ax.set_title('Tukey lambda PPCC Plot')

    refine_plot(ax, fontsize=20)
    # plt.show()
    plt.savefig('figure/BF_analysis/band_{}_read_PPCC.png'.format(k), bbox_inches='tight')        
    plt.clf()

    ax = plt.subplot(111)
    _, (scale, loc, r) = stats.probplot(x, best_shape_val, plot=ax, dist='tukeylambda')        
    ax.set_title('Tukey lambda Probability Plot')

    refine_plot(ax, r, fontsize=20)

    # plt.show()
    plt.savefig('figure/BF_analysis/band_{}_read_TL.png'.format(k), bbox_inches='tight')        
    plt.clf()


def outlier_correct(frame, p=0.01):
    # percentile = np.percentile(frame, [1, 99])
    percentile = np.percentile(frame, [p, 100-p])
    print(percentile)
    # bound = np.abs(percentile).max()
    frame[frame < percentile[0]] = percentile[0]
    frame[frame > percentile[1]] = percentile[1]
    # frame[frame < percentile[0]] = -bound
    # frame[frame > percentile[1]] = bound
    return frame


def banding_FFT(frame, k=128):
    def robust_mean(data, percent):
        p = np.percentile(data, [percent, 100-percent])
        ind = np.logical_and(data > p[0], data < p[1])
        return data[ind].mean()

    if frame.ndim == 3:
        frame = frame[0, ...]

    fimg = np.fft.fft2(frame, norm='ortho')
    fimg = np.fft.fftshift(np.abs(fimg))
    fimg = outlier_correct(fimg)


    fimg = crop_center(fimg[None], 128, 128)[0,...]
    # fimg = crop_center(fimg[None], 512, 512)[0,...]

    plt.imshow(fimg)
    plt.colorbar()
    plt.grid(False)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    # plt.show() 
    plt.savefig('figure/BF_analysis/band_{}_FFT.png'.format(k), bbox_inches='tight')        
    plt.clf()


def FF_analysis(k, flat_frame_1, flat_frame_2):
    # cropx = 13; cropy = 13
    
    # Image1 = io.imread('/media/zhangtao/UUI/HSI/color/2021-01-19-1447_I99.8_1.tif').astype(np.float32)
    # Image2 = io.imread('/media/zhangtao/UUI/HSI/color/2021-01-19-1452_I99.8_3.tif').astype(np.float32)
    # print(Image1.shape)
    dn_signals = []
    signal_vars = []

    save_path = 'K/color_{}.npy'.format(k)

    # if os.path.exists(save_path):
    if False:
        res = np.load(save_path, allow_pickle=True).item()
        dn_signals = res['dn_signals']
        signal_vars = res['signal_vars']
    else:
        # for x in range(162, 450, 77):
        #     for y in range(162, 550, 77):
        #         print('processing: {}, {}'.format(x, y))
        #         img1 = np.rot90(Image1[k])
        #         img2 = np.rot90(Image2[k])
        #         img1 = img1[x-cropx:x+cropx, y-cropy:y+cropy]
        #         img2 = img2[x-cropx:x+cropx, y-cropy:y+cropy]
        #         # cv2.imshow('1', img1/np.amax(Image1[k]))
        #         # cv2.imshow('2', img2/np.amax(Image2[k]))
        #         # cv2.imshow('3', Image2[k]/np.amax(Image2[k]))
        #         # cv2.waitKey()
                # noise_var = ((img1 - img2)/np.sqrt(2)).var()

                # signal = np.median((img1 + img2)/2)

                # dn_signals.append(signal)
                # signal_vars.append(noise_var)

        for i in range(len(flat_frame_1)):
            data1 = flat_frame_1[i][k]
            data2 = flat_frame_2[i][k]

            noise_var = ((data1 - data2)/np.sqrt(2)).var()
            signal = np.median((data1 + data2)/2)

            dn_signals.append(signal)
            signal_vars.append(noise_var)

        print(len(dn_signals))
        # print(signal_vars)
        dn_signals = np.array(dn_signals)[::-1]
        signal_vars = np.array(signal_vars)[::-1]
    diff = signal_vars[1:] - signal_vars[:-1]
    print(dn_signals)
    print(diff)
    # neg_pos = np.nonzero(diff > 0)[0]
    # print(neg_pos)
    # if len(neg_pos) > 1:
    #     if neg_pos[1] - neg_pos[0] > 2:
    #         N = neg_pos[1]+1
    #     else:
    #         N = neg_pos[0]+1
    # else:
    #     N = neg_pos[0]+1
    # N = neg_pos[-1]+2
    N = len(dn_signals)
    print(N)
    dn_signals = dn_signals[:N]
    signal_vars = signal_vars[:N]

    slope, intercept, r, prob, sterrest = stats.linregress(dn_signals, signal_vars)
    
    if not os.path.exists(save_path):
        np.save(save_path, {'dn_signals': dn_signals, 'signal_vars': signal_vars, 'K': slope})
    
    N = len(signal_vars)
    signal_vars_est = dn_signals * slope + intercept
    rss = np.sum((signal_vars - signal_vars_est)**2)
    se = np.sqrt(rss / (N - 2))
    
    sx = np.sqrt(np.sum((dn_signals - dn_signals.mean())**2) / N)
    slope_se = se / (sx * np.sqrt(N))

    ax = plt.subplot(111)
    plt.plot(dn_signals, signal_vars, 'bo', dn_signals, signal_vars_est, 'r-')
    xmin = np.amin(dn_signals)
    xmax = np.amax(dn_signals)
    ymin = np.amin(signal_vars)
    ymax = np.amax(signal_vars)
    posx = xmin + 0.60 * (xmax - xmin)
    posy = ymin + 0.01 * (ymax - ymin)
    
    plt.text(posx, posy, '$K=%1.4f$' %(slope), fontsize=18 )
    
    # plt.text(posx, posy, '$\sigma_R=%1.4f DN$' %(np.sqrt( intercept )))
    plt.xlabel('estimated signal level (DN)')
    plt.ylabel('variance (DN)')
    
    refine_plot(ax)
    plt.savefig('figure/FF_analysis/band_{}.png'.format(k), bbox_inches='tight')
    print('Plot saved in band_{}.png'.format(k))
    plt.clf()
    return slope, slope_se

if __name__ == '__main__':
    if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/figure/BF_analysis'):
        os.makedirs('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/figure/BF_analysis')
    for k in range(25):
        # FF_analysis(k)
        BF_ISO_analysis(k)

    # if not os.path.exists('K'):
    #     os.mkdir('K')
    # if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/figure/FF_analysis'):
    #     os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/figure/FF_analysis')
    
    # ff_list = ['25', '50', '75', '100', '125', '150', '175']
    # flat_frame_1 = []
    # flat_frame_2 = []
    # for i in ff_list:
    #     flat_frame_1.append(np.load(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/flatframe', 'ff_'+i+'_1.npy'))[:, 109-50:109+50, 205-50:205+50])
    #     flat_frame_2.append(np.load(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/flatframe', 'ff_'+i+'_2.npy'))[:, 109-50:109+50, 205-50:205+50])

    # ff_list = ['25', '50', '75', '100', '150', '200']
    # flat_frame_1 = []
    # flat_frame_2 = []
    # for i in ff_list:
    #     flat_frame_1.append(np.load(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/flatframe_v5', 'ff_'+i+'_1.npy'))[:, 109-50:109+50, 205-50:205+50])
    #     flat_frame_2.append(np.load(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/flatframe_v5', 'ff_'+i+'_2.npy'))[:, 109-50:109+50, 205-50:205+50])    

    # for k in range(25):
    #     FF_analysis(k, flat_frame_1, flat_frame_2)

    # for camera, suffix in zip(['x1', 'wide'], ['.raw']*2):
    #     print('----------- {} -----------'.format(camera))

    #     basedir = 'dataset/FlatField/{}'.format(camera)
    #     for iso in [800, 1600, 3200]:
    #         print('------ {} ------'.format(iso))
    #         FF_analysis(basedir, str(iso), suffix) # Flatfiled frame analysis

    #     basedir = 'dataset/BiasFrames/{}'.format(camera)
    #     BF_ISO_analysis(basedir, suffix) # Bias frame analysis
