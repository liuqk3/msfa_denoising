import numpy as np
from spectral import *
import array
import cv2
from os.path import join
import os
from PIL import Image
from natsort import natsorted
# from pixel_correction import *

# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')

blacklevel=32

# with open('/home/jiangyuqi/Desktop/demosaicing/demosaicing/unprocessed/frame.raw', 'rb') as f:
# # with open('/home/jiangyuqi/Desktop/demosaicing/demosaicing/processed/demosaicing.raw', 'rb') as f:
#     # data = array.array('b')
#     # data.fromfile(f, 1085 * 2045 * 4)
#     # data = np.fromfile(f)
#     # npArray = np.frombuffer(data.tobytes(), dtype='<f4')
#     data = f.read()
#     npArray = np.frombuffer(data, dtype='<f4')
#     npArray = npArray / 1023.0
#     npArray.shape = (1085, 2045)

#     cv2.imshow('img', npArray)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

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

def show_img(img):
    img = img / 1023* 255
    img = img
    img = np.clip(img, 0, 255)
    # from PIL import Image
    Image.fromarray(img[0].astype(np.uint8)).save('test.png')

def get_avg_results(path, scene_num, avg_num=10, expo=800):
    img_avg = None
    for i in range(avg_num):
        img_path = join(path, 'scene_'+str(scene_num)+'_'+str(expo)+'_'+str(i+1))
        img = read_file(join(img_path, 'unprocessed','frame.raw')) / 1023
        if img_avg is None:
            img_avg = np.zeros_like(img)
        img_avg = img_avg + img

        print(np.mean(img))

    img_avg = img_avg / avg_num

    return img_avg

def generate_scene_data():
    path = '/home/jiangyuqi/Desktop/HSI_data'
    save_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v4'
    save_img_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_img_v4'
    scene_dir = []

    deprecated_list_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/deprecated_data_list.txt'
    deprecated_list = []
    with open(deprecated_list_path) as f:
        line = f.readline()
        while line != '':
            deprecated_list.append(line[:-1])
            line = f.readline()

    for dir in os.listdir(path):
        if dir[:5] != 'scene':
            continue
        for sub_dir in os.listdir(join(path, dir)):
            loc = sub_dir.find('_', 6)
            if sub_dir[:loc] in deprecated_list:
                continue
            scene_dir.append(join(path, dir, sub_dir, 'unprocessed', 'frame.raw'))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)

    scene_dir = natsorted(scene_dir)

    record_dic = {}

    ratio_table = {'800':'gt', '80':'x10', '40':'x20', '20':'x40'}

    for dir in scene_dir:
        dir_split = (dir.split('/')[6]).split('_')
        if not os.path.exists(join(save_path,'scene_'+dir_split[1])):
            os.mkdir(join(save_path,'scene_'+dir_split[1]))
            os.mkdir(join(save_img_path,'scene_'+dir_split[1]))
            os.mkdir(join(save_img_path,'scene_'+dir_split[1], 'gt'))
            os.mkdir(join(save_img_path,'scene_'+dir_split[1], 'x10'))
            os.mkdir(join(save_img_path,'scene_'+dir_split[1], 'x20'))
            os.mkdir(join(save_img_path,'scene_'+dir_split[1], 'x40'))

        data = read_file(dir)
        data = data / 1023
        data = np.clip(data, 0, 1)
        if len(dir_split) == 4:
            if dir_split[1] not in record_dic.keys():
                record_dic[dir_split[1]] = data
            else:
                record_dic[dir_split[1]] = data + record_dic[dir_split[1]]
        else:
            print(dir)
            np.save(join(save_path,'scene_'+dir_split[1], dir_split[2]+'.npy'), data)

            img = np.transpose(data, (1,2,0))
            img = img * 255 * 800 / int(dir_split[2])
            img = np.clip(img, 0, 255).astype(np.uint8)
            for i in range(25):
                Image.fromarray(img[:, :, i]).save(join(save_img_path,'scene_'+dir_split[1],\
                    ratio_table[dir_split[2]],'band_'+str(i)+'.png'))

            color_img = np.stack((img[:,:,0],img[:,:,12], img[:, :, 24]),axis=2)
            Image.fromarray(color_img).save(join(save_img_path,'scene_'+dir_split[1],\
                    ratio_table[dir_split[2]],'rgb.png'))


    for key in record_dic.keys():
        data = record_dic[key] / 50
        np.save(join(save_path,'scene_'+key, '800.npy'), data)
        img = np.transpose(data, (1,2,0))
        img = np.clip(img*255, 0, 255).astype(np.uint8)
        for i in range(25):
            Image.fromarray(img[:, :, i]).save(join(save_img_path,'scene_'+key,\
                    'gt','band_'+str(i)+'.png'))

        color_img = np.stack((img[:,:,0],img[:,:,12], img[:, :, 24]),axis=2)
        Image.fromarray(color_img).save(join(save_img_path,'scene_'+key,\
                    'gt','rgb.png'))


def generate_flat_frame():
    save_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/flatframe_v5'
    save_img_path = save_path.replace('flatframe_v5', 'flatframe_img_v5')
    load_path = '/home/jiangyuqi/Desktop/HSI_data/flatframe_v5'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)

    ratio_list = ['1', '01', '002', '5', '25', '50', '75', '100', '125', '150', '175', '200']

    for r in ratio_list:
        for i in range(1,6):
            ff_path = join(load_path, 'ff_'+r+'_'+str(i), 'unprocessed', 'frame.raw')
            data = read_file(ff_path)
            np.save(join(save_path, 'ff_'+r+'_'+str(i)+'.npy'), data)

            print(np.mean(data))

            Image.fromarray((np.clip(data/1023*255,0,255))[0][109-50:109+50, 205-50:205+50].astype(np.uint8)).save(join(save_img_path, 'ff_'+r+'_'+str(i)+'.png'))



if __name__ == "__main__":
    generate_scene_data()
    # generate_flat_frame()

# if __name__ == "__main__":
#     path ='/home/jiangyuqi/Desktop/HSI_data/biasframe_1k'
#     dirs = os.listdir(path)

#     for d in dirs:
#         bf_path = join(path, d, 'unprocessed', 'frame.raw')
#         print(np.mean(read_file(bf_path)))

# if __name__ == "__main__":
#     print('Processing bias frame ...')
#     if not os.path.exists('data/biasframe_v2'):
#         os.mkdir('data/biasframe_v2')
#     for i in range(1, 21):
#     # for i in [6]:
#         path = '/home/jiangyuqi/Desktop/bias_frame_02'
#         data = read_file(join(path, 'bf_'+str(i), 'unprocessed', 'frame.raw'))
#         # data = data[:, 10:197, 10:399]
#         # print(np.mean(data[:,10:207,10:399]))
#         np.save(join('data/biasframe_v2', 'bf_'+str(i)), data)
#         # Image.fromarray((data[0]*10*255).astype(np.uint8)).save(join('data/biasframe_img', 'bf_'+str(i)+'.png'))

#     print('Processing flat frame ...')
#     if not os.path.exists('data/flatframe'):
#         os.mkdir('data/flatframe')
#     # if not os.path.exists('data/flatframe_img'):
#     #     os.mkdir('data/flatframe_img')
#     ff_list = ['001', '01', '1', '25', '50', '75', '100', '125', '150', '175']
#     mean_val_list = []
#     for i in ff_list:
#     # for i in [6]:
#         path = '/home/jiangyuqi/Desktop/flatframe'
#         data = read_file(join(path, 'ff_'+i+'_1', 'unprocessed', 'frame.raw'))
#         np.save(join('data/flatframe', 'ff_'+i+'_1'), data)
#         # data = data[:, 109-50:109+50, 205-50:205+50]
#         # Image.fromarray((data[0]*255).astype(np.uint8)).save(join('calibration/flatframe_img', 'ff_'+i+'_1'+'.png'))

#         data = read_file(join(path, 'ff_'+i+'_2', 'unprocessed', 'frame.raw'))
#         np.save(join('data/flatframe', 'ff_'+i+'_2'), data)
#         # data = data[:, 109-50:109+50, 205-50:205+50]
#         # Image.fromarray((data[0]*255).astype(np.uint8)).save(join('calibration/flatframe_img', 'ff_'+i+'_2'+'.png'))
#         # mean_val_list.append(mean_val)
#     # ff_list = ['0.01', '0.1', '1', '25', '50', '75', '100', '125', '150', '175']
#     # plt.scatter(ff_list, mean_val_list)
#     # plt.show()
    
#     # path = '/home/jiangyuqi/Desktop/testframe4'
#     # # for i in range(1, 11):
#     # #     data = read_file(join(path, 'test'+str(i), 'unprocessed', 'frame.raw'))
#     # #     Image.fromarray((data[0]*255).astype(np.uint8)).save('test'+str(i)+'.png')
#     # data = read_file(join(path, 'test1', 'unprocessed', 'frame.raw'))
#     # Image.fromarray((data[0]*255).astype(np.uint8)).save('test1.png')
#     # data = read_file(join(path, 'test2', 'unprocessed', 'frame.raw'))
#     # Image.fromarray((np.clip(data[0]*10*255,0,255)).astype(np.uint8)).save('test2.png')    
#     # data = read_file(join(path, 'test3', 'unprocessed', 'frame.raw'))
#     # Image.fromarray((np.clip(data[0]*20*255,0,255)).astype(np.uint8)).save('test3.png')

#     # # path = '/home/jiangyuqi/Desktop/testframe2'
#     # data = read_file(join(path, 'test4', 'unprocessed', 'frame.raw'))
#     # Image.fromarray((data[0]*40*255).astype(np.uint8)).save('test4.png')
#     # # data = read_file(join(path, 'test2', 'unprocessed', 'frame.raw'))
#     # # Image.fromarray((np.clip(data[0]*2*255,0,255)).astype(np.uint8)).save('test_2.png')

#     path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data'

#     # path = '/home/jiangyuqi/Desktop/scene_0101'
#     # data1 = read_file(join(path, 'scene_31_800', 'unprocessed', 'frame.raw'))
#     # # data2 = read_file(join(path, 'scene_31_800_mosaic', 'unprocessed', 'frame.raw'))

#     # Image.fromarray((np.clip(data1*255,0,255)).astype(np.uint8)).show()
#     # Image.fromarray((np.clip(data2[0]*255,0,255)).astype(np.uint8)).show()

#     # print(np.mean(data1-data2))

#     # for i in range(74):
#     #     print('Process scene '+ str(i+1)+ ' ...')
#     #     if i != 5:
#     #         gt_expo = read_file(join(path, 'scene_'+str(i+1)+'_'+'800', 'unprocessed', 'frame.raw'))
#     #         x10_expo = read_file(join(path, 'scene_'+str(i+1)+'_'+'80', 'unprocessed', 'frame.raw'))
#     #         x20_expo = read_file(join(path, 'scene_'+str(i+1)+'_'+'40', 'unprocessed', 'frame.raw'))
#     #         x40_expo = read_file(join(path, 'scene_'+str(i+1)+'_'+'20', 'unprocessed', 'frame.raw'))
#     #     else:
#     #         gt_expo = read_file(join(path, 'scene_'+str(i+1)+'_'+'500', 'unprocessed', 'frame.raw'))
#     #         x10_expo = read_file(join(path, 'scene_'+str(i+1)+'_'+'50', 'unprocessed', 'frame.raw'))
#     #         x20_expo = read_file(join(path, 'scene_'+str(i+1)+'_'+'25', 'unprocessed', 'frame.raw'))
#     #         x40_expo = read_file(join(path, 'scene_'+str(i+1)+'_'+'1205', 'unprocessed', 'frame.raw'))
        
#     #     gt = np.stack([gt_expo[0,:, :], gt_expo[13, :, :], gt_expo[24, :, :]])
#     #     x10 = np.stack([x10_expo[0, :, :], x10_expo[13, :, :], x10_expo[24, :, :]])
#     #     x20 = np.stack([x20_expo[0, :, :], x20_expo[13, :, :], x20_expo[24, :, :]])   
#     #     x40 = np.stack([x40_expo[0, :, :], x40_expo[13, :, :], x40_expo[24, :, :]])     

#     #     gt = np.transpose(gt, (1,2,0))
#     #     x10 = np.transpose(x10, (1,2,0))
#     #     x20 = np.transpose(x20, (1,2,0))
#     #     x40 = np.transpose(x40, (1,2,0))
#     #     # gt = gt_expo
#     #     # x10 = x10_expo
#     #     # x20 = x20_expo
#     #     # x40 = x40_expo

#     #     Image.fromarray((gt[:,:,0]*255).astype(np.uint8)).save('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_img/scene_'+str(i+1)+'_gt.png')
#     #     Image.fromarray((np.clip(x10*10*255, 0, 255)).astype(np.uint8)).save('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_img/scene_'+str(i+1)+'_x10.png')
#     #     Image.fromarray((np.clip(x20*20*255, 0, 255)).astype(np.uint8)).save('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_img/scene_'+str(i+1)+'_x20.png')
#     #     Image.fromarray((np.clip(x40*40*255, 0, 255)).astype(np.uint8)).save('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_img/scene_'+str(i+1)+'_x40.png')

#     # path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_data'
#     # if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_img'):
#     #     os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_img')

#     # for i in range(74):
#     #     img = np.load(join(path, 'data_'+str(i+1)+'.npy'))
#     #     img = np.transpose(img, (1,2,0))
#     #     Image.fromarray((img[:, :, 0]*255).astype(np.uint8)).save('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_img/scene_'+str(i+1)+'_gt.png')


#     # for i in range(74):
#     #     path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data'
#     #     img1 = np.array(Image.open(join(path, 'corrected_img','scene_'+str(i+1)+'_gt.png')))
#     #     img2 = np.array(Image.open(join(path, 'scene_img','scene_'+str(i+1)+'_gt.png')))
#     #     Image.fromarray(np.abs(img1-img2)).save('test'+str(i+1)+'.png')