from calendar import c
import numpy as np
# import read
from os.path import join
from matplotlib import pyplot as plt
import scipy.stats as stats
from PIL import Image
import os

from torch import batch_norm_gather_stats
from read import read_file
import xml.etree.ElementTree as ET
from read import read_file

path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data'

def bf_npy2img():
    for i in range(20):
        bf_path = join(path, 'biasframe_v2', 'bf_'+str(i+1)+'.npy')
        data = np.load(bf_path)
        data = data / 1023 * 255*100
        data = np.transpose(data, (1,2,0))
        data = np.clip(data, 0, 255)

        if not os.path.exists(join(path, 'biasframe_img_v2', 'bf_'+str(i+1))):
            os.mkdir(join(path, 'biasframe_img_v2', 'bf_'+str(i+1)))

        for j in range(25):
            Image.fromarray(data[:, :, j].astype(np.uint8)).save(join(path, 'biasframe_img_v2', 'bf_'+str(i+1), 'band_'+str(j+1)+'.png'))



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
        plt.titel('Probability Plot ($R^2 = {:.3f}$)'.format(r**2))

def cal_col_mean():
    bf_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/biasframe_v2'    
    index = []
    mean_val = []
    for i in range(20):
        data = np.load(join(bf_path, 'bf_'+str(i+1)+'.npy'))
        print(np.mean(data))
        # mean_val

        if np.mean(data) < 3.8 or np.mean(data) > 4.2:
            index.append(i)
    print(index)
    # result = [1, 2, 7, 8, 9, 10, 12, 13, 14, 15, 17, 19]
        # for k in range(25):
        #     print(np.mean(data[k]))

def fit_colorbias():
    bf_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/biasframe_v2'    
    color_bias_mean = []
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

def plot_col():
    ana_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/biasframe_analyze_v3'

    if not os.path.exists(ana_path):
        os.mkdir(ana_path)

    bf_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/biasframe_v2'

    # col_mean = []
    # col_var = []
    # row_mean = []
    # row_var = []

    # min_i = 0
    # min_j = 0
    # min_var = 10000000

    for i in range(20):
        data = np.load(join(bf_path, 'bf_'+str(i+1)+'.npy'))

        # if not os.path.exists(join(ana_path, 'bf_'+str(i+1))):
        #     os.mkdir(join(ana_path, 'bf_'+str(i+1)))

        plt.figure()
        c = {8:'b', 16:'y', 24:'r'}
        label = {8:'892nm', 16:'741nm', 24:'713nm'}
        # for j in range(25):
        for j in [8,16,24]:
            img = data[j]

            #col:
            x = img.mean(axis=0, keepdims=True)
            print(x.shape)
            # plt.figure()
            plt.plot(range(x.shape[1]),x[0],color=c[j], label=label[j])
        plt.legend(fontsize=18, loc=1)
        plt.xlabel('Column Coordinate',fontsize=20)
        plt.ylabel('Mean Value of Each Column',fontsize=20)
        plt.ylim((-2, 8))
        # import pdb; pdb.set_trace()
        plt.savefig(join(ana_path, 'col_'+str(i)+'.pdf'))

        plt.clf()

            # col_mean.append(np.mean(x))
            # col_var.append(np.var(x))

        plt.figure()
        # for j in range(25):
        for j in [8,16,24]:
            #row:
            img = data[j]
            x = img.mean(axis=1)[2:-5]
            print(x.shape)
            # plt.figure()

            plt.plot(range(x.shape[0]),x,color=c[j], label=label[j])
        plt.xlabel('Row Coordinate',fontsize=20)
        plt.ylabel('Mean Value of Each Row',fontsize=20)
        plt.ylim((-2, 8))
        plt.legend(fontsize=18,loc=1)
        plt.savefig(join(ana_path, 'row_'+str(i)+'.pdf'))


            # if np.var(x) < min_var:
            #     min_i = i
            #     min_j = j
            #     min_var = np.var(x)

        plt.clf()
            # row_mean.append(np.mean(x))
            # row_var.append(np.var(x))
    # print('row mean:')
    # print(row_mean)
    # print('row var:')
    # print(np.mean(row_var))
    # # print('col mean:')
    # # print(col_mean)
    # print('col var:')
    # print(np.mean(col_var))

    # print(min_i, min_j)

    
def check_bf1k():
    path = '/home/jiangyuqi/Desktop/HSI_data/biasframe_1k'
    

    cnt_array = np.array([0,0,0,0,0,0,0,0])

    total_val = 0
    val = 0

    for i in range(1, 601):
    # for i in range(1, 21):
        bf_path = join(path, 'bf_'+str(i), 'unprocessed', 'frame.raw')
        data = read_file(bf_path)

        mean_val = np.mean(data)

        total_val += mean_val

        if mean_val < 4:
            val += mean_val


        for j in range(8):
            if mean_val < j+1:
                cnt_array[j] += 1
                break

    print(cnt_array)
    print(total_val / 600)
    print(val / (cnt_array[2]+cnt_array[3]))

    total_val = 0
    cnt_array = np.array([0,0,0,0,0,0,0,0])


    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/biasframe_v2'
    for i in range(1, 21):
        data = np.load(join(path, 'bf_'+str(i) + '.npy'))
        mean_val = np.mean(data[:, 109-70:109+70, 205-70:205+70])
        total_val += mean_val


        for j in range(8):
            if mean_val < j+1:
                cnt_array[j] += 1
                break        

    print(cnt_array)
    print(total_val / 20)

def check_bf_T():
    path = '/home/jiangyuqi/Desktop/HSI_data/biasframe_1k'

    mean_val_list = []
    T_list = []

    for i in range(1, 601):
    # for i in range(1, 21):
        bf_path = join(path, 'bf_'+str(i), 'unprocessed', 'frame.raw')
        data = read_file(bf_path)

        mean_val = np.mean(data)

        xml_file = ET.parse(bf_path + '.xml')
        temperature = xml_file.getroot()[1][0][10].attrib['value']

        mean_val_list.append(mean_val)
        T_list.append(temperature)

    print(mean_val_list)
    print(T_list)

    plt.figure()
    plt.scatter(T_list, mean_val_list)
    plt.show()


def check_scene_T():
    path = '/home/jiangyuqi/Desktop/HSI_data'
    scene_dir = []
    for dir in os.listdir(path):
        if dir[:5] != 'scene':
            continue
        for sub_dir in os.listdir(join(path, dir)):
            scene_dir.append(join(path, dir, sub_dir, 'unprocessed', 'frame.raw'))


    ratio_table = {'800':'gt', '80':'x10', '40':'x20', '20':'x40'}

    expo800_T_list = []
    expo80_T_list = []
    expo40_T_list = []
    expo20_T_list = []

    for dir in scene_dir:
        #dir_split:
        #'scene', 'scene_number', 'expo', '800expo_number'

        dir_split = (dir.split('/')[6]).split('_')
        xml_file = ET.parse(dir + '.xml')
        temperature = xml_file.getroot()[1][0][10].attrib['value']

        if dir_split[2] == '800':
            expo800_T_list.append(temperature)
        elif dir_split[2] == '80':
            expo80_T_list.append(temperature)
        elif dir_split[2] == '40':
            expo40_T_list.append(temperature)
        else:
            expo20_T_list.append(temperature)

    print(expo800_T_list.sort())
    print(expo80_T_list.sort())
    print(expo40_T_list.sort())
    print(expo20_T_list.sort())

def bf_all_expo():
    dir = ['bf_1_test', 'bf_test_01', 'bf_test_05', 'bf_test_005', 'bf_test_02']

    for d in dir:
        mean_val = 0
        for i in range(20):
            path = join('/home/jiangyuqi/Desktop/HSI_data', d, 'bf_'+str(i+1), 'unprocessed', 'frame.raw')
            data = read_file(path)
            mean_val += np.mean(data)
        print(d)
        print(mean_val/20)

def cal_cb(gt, noise, ratio):
    return (noise * ratio - gt)/(ratio)

def ff_cal_cb():
    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/flatframe'
    ratio_table = {'01':0.1, '001':0.01}

    ratio_list = ['1', '01', '001', '25', '50', '75', '100', '125', '150']
    gt = (np.load(join(path, 'ff_175_1.npy'))+np.load(join(path, 'ff_175_2.npy')))/2

    cb = {}

    for r in ratio_list:
        noise = (np.load(join(path, 'ff_'+r+'_1.npy')) + np.load(join(path, 'ff_'+r+'_2.npy')))/2

        print(np.mean(noise))

        if r in ratio_table.keys():
            ratio = ratio_table[r]
        else:
            ratio = float(r)

        cb[r] = cal_cb(np.mean(gt), np.mean(noise), 175/ratio)

    print(cb)

def ff_cal_cb_v2():
    path = '/home/jiangyuqi/Desktop/HSI_data/flatframe_v5'

    ratio_table = {'01':0.1, '002':0.2}

    # gt = []
    # for i in range(1,2):
    #     gt.append(read_file(join(path, 'ff_400_'+str(i), 'unprocessed', 'frame.raw')))

    # gt = np.mean(gt, axis=0)

    ratio_list = ['1', '01', '002', '5', '25', '50', '75', '100', '125', '150', '175', '200']

    cb = {}

    for r in ratio_list:
        n = []
        for i in range(1,6):
            noise = read_file(join(path, 'ff_'+r+'_'+str(i), 'unprocessed', 'frame.raw'))
            n.append(noise)
        n = np.mean(n,axis=0)
        print(np.mean(n))


    for r in ratio_list:
        cb[r] = []

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

            noise = read_file(join(path, 'ff_'+r+'_'+str(i), 'unprocessed', 'frame.raw'))
            gt = read_file(join(path, 'ff_200_'+str(i), 'unprocessed', 'frame.raw'))

            # print(r)
            # print(np.mean(noise))
            # print(np.mean(gt))

            cb[r].append(cal_cb(np.mean(gt), np.mean(noise), 200/ratio))

    print(cb)
    cb_mean = {}
    for k in cb.keys():
        cb_mean[k] = np.mean(cb[k])
    
    print(cb_mean)



def calculate_cb_v1(gt_mean, noisy_path, ratio):
    
    noisy_mean = np.mean(read_file(noisy_path))

    color_bias = (noisy_mean * ratio-gt_mean)/(ratio)

    return color_bias

def check_color_bias_v1():
    # path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v2'
    path = '/home/jiangyuqi/Desktop/HSI_data/scene-0116-1'
    x10_cb = []
    x20_cb = []
    x40_cb = []

    for i in range(1, 176):
        data = []
        # for j in range(1,51):
        #     gt_path = join(path, 'scene_'+str(i)+'_800_'+str(j), 'unprocessed', 'frame.raw')

        #     data.append(read_file(gt_path))
        # # data = np.array(data) / 50
        # data = np.mean(data,axis=0)
        if not os.path.exists(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v4', 'scene_'+str(i), '800.npy')):
            continue
        data = np.load(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v4', 'scene_'+str(i), '800.npy'))
        data = np.mean(data)*1023

        # x10_path = join(path, 'scene_'+str(i)+'_80', 'unprocessed', 'frame.raw')
        # x20_path = join(path, 'scene_'+str(i)+'_40', 'unprocessed', 'frame.raw')
        # x40_path = join(path, 'scene_'+str(i)+'_20', 'unprocessed', 'frame.raw')
        
        # x10_cb.append(calculate_cb_v1(data, x10_path, 10))
        # x20_cb.append(calculate_cb_v1(data, x20_path, 20))
        # x40_cb.append(calculate_cb_v1(data, x40_path, 40))
        x10 = np.load(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v4', 'scene_'+str(i), '80.npy'))
        x20 = np.load(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v4', 'scene_'+str(i), '40.npy'))
        x40 = np.load(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v4', 'scene_'+str(i), '20.npy'))

        x10_cb.append(cal_cb(data, np.mean(x10)*1023, 10))
        x20_cb.append(cal_cb(data, np.mean(x20)*1023, 20))
        x40_cb.append(cal_cb(data, np.mean(x40)*1023, 40))

        # if cal_cb(data, np.mean(x20)*1023, 20) > 4:
        #     print(i)

    x10_cb.sort(reverse=True)
    x20_cb.sort(reverse=True)
    x40_cb.sort(reverse=True)
    print(np.mean(np.array(x10_cb)))
    # print(np.std(np.array(x10_cb)))
    print(np.mean(np.array(x20_cb)))
    # print(np.std(np.array(x20_cb)))
    print(np.mean(np.array(x40_cb)))
    # print(np.std(np.array(x40_cb)))

    # print(x10_cb[:10])
    # print(x20_cb[:10])
    # print(x40_cb[:10])





def calculate_cb_v2(gt_path, noisy_path, ratio):
    
    noisy_mean = np.mean(read_file(noisy_path))
    gt_mean = np.mean(read_file(gt_path))

    color_bias = (noisy_mean * ratio-gt_mean)/(ratio-1)

    return color_bias


def check_color_bias_v2():
    # path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v2'
    path = '/home/jiangyuqi/Desktop/HSI_data/scene-0116-1'
    x10_cb = []
    x20_cb = []
    x40_cb = []

    for i in range(1, 146):
        # print('scene'+str(i))
        # for j in range(50):
        #     print(np.mean(read_file(join(path, 'scene_'+str(i)+'_800_'+str(j+1), 'unprocessed', 'frame.raw'))))

        gt_path = join(path, 'scene_'+str(i)+'_800_50', 'unprocessed', 'frame.raw')
        x10_path = join(path, 'scene_'+str(i)+'_80', 'unprocessed', 'frame.raw')
        x20_path = join(path, 'scene_'+str(i)+'_40', 'unprocessed', 'frame.raw')
        x40_path = join(path, 'scene_'+str(i)+'_20', 'unprocessed', 'frame.raw')
        # print(np.mean(read_file(x40_path))*40)
        
        x10_cb.append(calculate_cb_v2(gt_path, x10_path, 10))
        x20_cb.append(calculate_cb_v2(gt_path, x20_path, 20))
        x40_cb.append(calculate_cb_v2(gt_path, x40_path, 40))

    print(np.mean(np.array(x10_cb)))
    print(np.mean(np.array(x20_cb)))
    print(np.mean(np.array(x40_cb)))


def check_scene_data():
    path = '/home/jiangyuqi/Desktop/HSI_data/scene-0116-1'

    cnt = np.zeros(33)

    for i in range(1,146):
        scene_list = []
        for j in range(1,51):
            scene_path = join(path, 'scene_'+str(i)+'_800_'+str(j), 'unprocessed', 'frame.raw')
            data = read_file(scene_path)
            scene_list.append(np.mean(data))

        print('scene'+str(i)+':')
        print(np.mean(scene_list))
        print(np.std(scene_list))
        print(np.std(scene_list)/np.mean(scene_list))
        print('\n')

        for k in range(0,34):
            if (k+1) * 0.03 > np.std(scene_list)/np.mean(scene_list):
                cnt[k] += 1
                break

    print(cnt)


def plot_cb_parameter():
    path ='/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_v3_exp'
    a_matrix = np.load(join(path, 'a_matrix.npy'))
    b_matrix = np.load(join(path, 'b_matrix.npy'))
    c_matrix = np.load(join(path, 'c_matrix.npy'))

    x = a_matrix+c_matrix
    print(x.shape)

    fig_path = path.replace('color_bias_v3_exp','bf_color_bias')
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    print(np.mean(x)*4)

    # for i in range(25):
    #     x_i = x[i]*4
    #     print(np.mean(x_i))
        # plt.figure()
        # plt.ylim((-5, 10))
        # index = np.array(range(x.shape[1]))
        # plt.plot(index, x_i)
        # plt.savefig(join(fig_path, str(i)+'.png'))
        # plt.clf()




def get_color_bias_from_bias_frame():

    def convert_cubic_image_to_one_channel_image(image):
        h, w = image.shape[1], image.shape[2]
        sep = 10
        image_tmp = np.zeros((5*h+4*sep, 5*w+4*sep)) + float('inf')
        for bidx in range(25):
            col = bidx % 5
            row = bidx // 5
            start_col = col * (w+sep)
            start_row = row * (h+sep)
            # import pdb; pdb.set_trace()
            image_tmp[start_row:start_row+h, start_col:start_col+w] = image[bidx]
        return image_tmp

    save_dir = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/calibration/color_bias_vis/based_on_bias_frame'
    os.makedirs(save_dir, exist_ok=True)

    bf_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/biasframe_v2'

    mean_frame = 0
    count = 20
    for i in range(count):
        one_frame = np.load(join(bf_path, 'bf_'+str(i+1)+'.npy'))
        one_frame = one_frame[:,4:,:]
        # import pdb; pdb.set_trace()
        mean_frame += one_frame
    
    mean_frame /= count
    mean_frame = (mean_frame - mean_frame.min())/(mean_frame.max() - mean_frame.min())
    
    scale = 3
    # save all band into one image 
    mean_frame_img = convert_cubic_image_to_one_channel_image(mean_frame)
    mean_frame_img = Image.fromarray((np.clip(mean_frame_img, 0, 1)*scale*255).astype(np.uint8))
    save_path = os.path.join(save_dir, 'mean_frame.png')
    mean_frame_img.save(save_path)

    # save each band
    for bidx in range(mean_frame.shape[0]):
        band = mean_frame[bidx]
        print('{}: mean: {}, min: {}, max: {}'.format(bidx, band.mean(), band.min(), band.max()))
        band_img = Image.fromarray((np.clip(band, 0, 1)*scale*255).astype(np.uint8))
        save_path = os.path.join(save_dir, 'mean_band_{}.png'.format(bidx))
        band_img.save(save_path)

       



if __name__ == "__main__":
    # bf_npy2img()
    # plot_col()
    # check_bf_T()
    # check_scene_T()
    # bf_all_expo()
    # check_bf1k()
    # check_color_bias_v1()
    # check_scene_data()
    # ff_cal_cb_v2()
    # plot_cb_parameter()
    # cal_col_mean()
    # fit_colorbias()
    get_color_bias_from_bias_frame()