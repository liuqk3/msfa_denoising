from dataclasses import replace
from logging.config import valid_ident
import numpy as np
import os
from os.path import join
from read import read_file
from PIL import Image
from copy import deepcopy


def dpc_v1(img,threshold = 0.1):
    # return the mask of dead pixel and the corrected img, the dead pixel is replaced by the median of the neighborhood
    # use the average value of neighborhood and the threshold to determine whether a pixel is a dead pixel
    mask = np.zeros_like(img)
    img_corrected = np.zeros_like(img)
    c, h, w = img.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                a = img[k, max(0,i-1):min(i+2,h), max(0,j-1):min(j+2,w)].reshape(-1)
                a.sort()
                # mean_val = np.mean(img[k, max(0,i-1):min(i+2,h), max(0,j-1):min(j+2,w)])
                # if np.abs(img[k,i,j]-mean_val)/mean_val > threshold:
                if img[k,i,j] > (a[-2]+threshold) or img[k,i,j] < (a[1]-threshold):
                    mask[k,i,j] = 1
                    img_corrected[k,i,j] = np.median(img[k, max(0,i-1):min(i+2,h), max(0,j-1):min(j+2,w)])
                else:
                    img_corrected[k, i, j] = img[k, i, j]

    return mask, img_corrected


def dpc_v2(img, rate=3):
    mask = np.zeros_like(img)
    img_corrected = np.zeros_like(img)
    c, h, w = img.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                a = np.array(img[k, max(0,i-1):min(i+2,h), max(0,j-1):min(j+2,w)]).reshape(-1)
                a.sort()
                sub_max = a[-2]
                sub_min = a[1]
                diff = sub_max - sub_min
                avg = (np.sum(a)-sub_max-sub_min-img[k, i, j])/(len(a)-3)

                if (avg-rate*diff)>img[k, i, j] or img[k, i, j]>(avg+rate*diff):
                    mask[k,i,j] = 1
                    img_corrected[k,i,j] = np.median(img[k, max(0,i-1):min(i+2,h), max(0,j-1):min(j+2,w)])
                else:
                    img_corrected[k, i, j] = img[k, i, j]


    return mask, img_corrected


def dpc_v3(img, rate=5):
    mask = np.zeros_like(img)
    img_corrected = np.zeros_like(img)
    c, h, w = img.shape

    for i in range(h):
        for j in range(w):
            for k in range(c):
                # a = np.array(img[k, max(0,i-1):min(i+2,h), max(0,j-1):min(j+2,w)]).reshape(-1)
                # a.sort()
                a = []
                for di in list(set([max(0,i-1),i,min(i+1,h-1)])):
                    for dj in list(set([max(0,j-1),j,min(j+1,w-1)])):
                        if di == i and dj == j:
                            continue
                        else:
                            a.append(img[k, di, dj])
                a = np.array(a)
                a.sort()
                sub_max = a[-2]
                sub_min = a[1]
                diff = sub_max - sub_min
                avg = (np.sum(a)-sub_max-sub_min)/(len(a)-2)

                if (avg-rate*diff)>img[k, i, j] or img[k, i, j]>(avg+rate*diff):
                    mask[k,i,j] = 1
                #     img_corrected[k,i,j] = np.median(a)
                # else:
                #     img_corrected[k, i, j] = img[k, i, j]


    return mask, img_corrected



def generate_mask():

    print('Generate mask using 175 expo flat frame...')

    mask_1, img_corrected, mean_ff = dpc_v3()
    np.save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask', 'mask_175'), mask_1)
    np.save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_data', 'corrected_ff_175'), img_corrected)

    img_corrected = np.transpose(img_corrected, (1,2,0))
    mean_ff = np.transpose(mean_ff, (1,2,0))

    
    Image.fromarray((img_corrected[:, :, 0] * 255).astype(np.uint8)).save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_img', 'corrected_img_175.png'))
    Image.fromarray((mean_ff[:, :, 0] * 255).astype(np.uint8)).save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_img', 'mean_img_175.png'))
    Image.fromarray((np.abs(mean_ff[:, :, 0]-img_corrected[:,:,0]) * 255).astype(np.uint8)).save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_img', 'sub_img_175.png'))
    Image.fromarray((mask_1[0]*255).astype(np.uint8)).save('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask/mask_175.png')

    print('Done!')
    print('Generate mask using 150 expo flat frame...')

    mask_2, img_corrected, mean_ff = dpc_v3(150)
    np.save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask', 'mask_150'), mask_2)
    np.save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_data', 'corrected_ff_150'), img_corrected)

    img_corrected = np.transpose(img_corrected, (1,2,0))
    mean_ff = np.transpose(mean_ff, (1,2,0))


    Image.fromarray((img_corrected[:, :, 0] * 255).astype(np.uint8)).save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_img', 'corrected_img_150.png'))
    Image.fromarray((mean_ff[:, :, 0] * 255).astype(np.uint8)).save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_img', 'mean_img_150.png'))
    Image.fromarray((np.abs(mean_ff[:, :, 0]-img_corrected[:,:,0]) * 255).astype(np.uint8)).save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_img', 'sub_img_150.png'))
    Image.fromarray((mask_2[0]*255).astype(np.uint8)).save('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask/mask_150.png')
    
    print('Done!')
    print('Combining two masks...')

    mask = mask_1 + mask_2

    c, h, w = mask.shape

    mask_loc = []

    for k in range(c):
        for i in range(h):
            for j in range(w):
                if mask[k, i, j] == 1:
                    mask_loc.append(np.array([k, i, j]))

    np.save('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask/mask.npy',np.array(mask_loc))
    print('Done!')

    return mask

def generate_mask_v2():
    print('Generate mask using mask of realistic scene data...')
    
    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v3'
    mask = np.zeros((25, 217, 409))
    tmp = np.zeros_like(mask)

    for i in range(175):

        print('dpc processing for scene'+str(i+1)+'...')

        if not os.path.exists(join(path, 'mask_'+str(i+1)+'.npy')):

            if not os.path.exists(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v3/', 'scene_'+str(i+1), '800.npy')):
                continue

            data = np.load(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v3/', 'scene_'+str(i+1), '800.npy'))
            data_mask, data_corrected = dpc_v2(data)

            np.save(join(path, 'mask_'+str(i+1)+'.npy'), data_mask)

    for i in range(175):

        if not os.path.exists(join(path, 'mask_'+str(i+1)+'.npy')):
            continue

        tmp_mask = np.load(join(path, 'mask_'+str(i+1)+'.npy'))
        tmp = tmp+tmp_mask

    if not os.path.exists(join(path, 'final_mask')):
        # os.mkdir(join(path, 'final_mask'))
        os.makedirs(join(path, 'final_mask'))

    mask = (tmp>75).astype(np.uint8)
    
    mask_img = np.transpose(mask, (1,2,0))
    
    for i in range(25):
        Image.fromarray(mask_img[:, :, i] * 255).save(join(path, 'final_mask', 'band_'+str(i)+'.png'))

    mask_loc = []
    c, h, w = mask.shape
    for k in range(c):
        for i in range(h):
            for j in range(w):
                if mask[k, i, j] == 1:
                    mask_loc.append(np.array([k, i, j]))
    np.save(join(path, 'final_mask.npy'), mask_loc)

    return mask_loc

def generate_mask_v3():
    print('Generate mask using mask of realistic scene data...')
    
    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v4'
    mask = np.zeros((25, 217, 409))
    tmp = np.zeros_like(mask)

    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(175):

        print('dpc processing for scene'+str(i+1)+'...')

        if not os.path.exists(join(path, 'mask_'+str(i+1)+'.npy')):

            if not os.path.exists(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v3/', 'scene_'+str(i+1), '800.npy')):
                continue

            data = np.load(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v3/', 'scene_'+str(i+1), '800.npy'))
            data_mask, data_corrected = dpc_v3(data)

            np.save(join(path, 'mask_'+str(i+1)+'.npy'), data_mask)

    for i in range(175):

        if not os.path.exists(join(path, 'mask_'+str(i+1)+'.npy')):
            continue

        tmp_mask = np.load(join(path, 'mask_'+str(i+1)+'.npy'))
        tmp = tmp+tmp_mask

    if not os.path.exists(join(path, 'final_mask')):
        # os.mkdir(join(path, 'final_mask'))
        os.makedirs(join(path, 'final_mask'))

    mask = (tmp>75).astype(np.uint8)
    
    mask_img = np.transpose(mask, (1,2,0))
    
    for i in range(25):
        Image.fromarray(mask_img[:, :, i] * 255).save(join(path, 'final_mask', 'band_'+str(i)+'.png'))

    mask_loc = []
    c, h, w = mask.shape
    for k in range(c):
        for i in range(h):
            for j in range(w):
                if mask[k, i, j] == 1:
                    mask_loc.append(np.array([k, i, j]))
    np.save(join(path, 'final_mask.npy'), mask_loc)

    return mask_loc


def correct_using_mask(data, mask):

    corrected_data = deepcopy(data)
    C, H, W = data.shape

    for i in range(len(mask)):
        c, h, w = mask[i]

        a = []
        for di in list(set([max(0,h-1),h,min(h+1,H-1)])):
            for dj in list(set([max(0,w-1),w,min(w+1,W-1)])):
                if di == h and dj == w:
                    continue
                else:
                    a.append(data[c, di, dj])
        a = np.array(a)

        # corrected_data[c, h, w] = np.median(data[c, max(0, h-1):min(H, h+2), max(0, w-1):min(W, w+2)])
        corrected_data[c, h, w] = np.median(a)

    return corrected_data

def correct_scene_data():
    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data'

    if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data'):
        os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data')

    if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_img'):
        os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_img')

    for i in range(74):

        print('Processing scene %d' % (i+1))

        if i+1 != 6:
            expo = '800'
        else:
            expo = '500'
        data_path = join(path, 'scene_'+str(i+1)+'_'+expo, 'unprocessed', 'frame.raw')
        data = read_file(data_path)

        mask = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask/mask.npy')

        corrected_data = correct_using_mask(data, mask)

        np.save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data', 'scene_'+str(i+1)+'_'+expo), data)

        img = np.transpose(corrected_data, (1,2,0))

        Image.fromarray((img[:,:,0]*255).astype(np.uint8)).save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_img', 'scene_'+str(i+1)+'_'+expo+'.png'))


def correct_scene_data_v2():
    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data'

    if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data_v2'):
        os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data_v2')

    # if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_img_v2'):
    #     os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_img_v2')   

    if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v2'):
        os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v2')  
    
    for i in range(74):
        print('Processing scene %d' % (i+1))

        if i+1 != 6:
            expo = '800'
        else:
            expo = '500'
        data_path = join(path, 'scene_'+str(i+1)+'_'+expo, 'unprocessed', 'frame.raw')
        data = read_file(data_path)

        mask, corrected_data = dpc_v2(data)

        img = np.transpose(corrected_data, (1,2,0))
        mask_img = np.transpose(mask, (1,2,0))

        if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data_v2/scene_'+str(i+1)):
            os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data_v2/scene_'+str(i+1))  
        if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v2/scene_'+str(i+1)):
            os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v2/scene_'+str(i+1))  

        np.save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data_v2/scene_'+str(i+1), 'data.npy'), corrected_data)
        np.save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v2/scene_'+str(i+1), 'data.npy'), mask)

        for j in range(25):
            Image.fromarray((img[:,:,j]*255).astype(np.uint8)).save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data_v2/scene_'+str(i+1), str(j)+'.png'))
            Image.fromarray((mask_img[:,:,j]*255).astype(np.uint8)).save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v2/scene_'+str(i+1), str(j)+'.png'))

def correct_scene_data_v3():
    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v4'

    if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v4'):
        os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v4')  
    
    generate_mask_v3()
    mask = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v4/final_mask.npy')

    if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data_v4'):
        os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data_v4')

    for i in range(175):
        print('Processing scene %d' % (i+1))

        expo = '800'
        expo_list = {'x40':'20', 'x20':'40', 'x10':'80'}

        # data_path = join(path, 'scene_'+str(i+1)+'_'+expo, 'unprocessed', 'frame.raw')
        # data = read_file(data_path)

        if not os.path.exists(join(path, 'scene_'+str(i+1), '800.npy')):
            continue

        data = np.load(join(path, 'scene_'+str(i+1), '800.npy'))

        corrected_data = correct_using_mask(data, mask)

        img = np.transpose(corrected_data, (1,2,0))
        # mask_img = np.transpose(mask, (1,2,0))

        # np.save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data_v3/clean_npy_file', 'scene_'+str(i+1)+'.npy'), corrected_data)
        np.save(join(path, 'scene_'+str(i+1), 'gt.npy'), corrected_data)

        if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data_v4/scene_'+str(i+1)):
            os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data_v4/scene_'+str(i+1))

        for j in range(25):
            Image.fromarray((img[:,:,j]*255).astype(np.uint8)).save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_scene_data_v4/scene_'+str(i+1), str(j)+'.png'))
            # Image.fromarray((mask_img[:,:,j]*255).astype(np.uint8)).save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v2/scene_'+str(i+1), str(j)+'.png'))


def correct_bias_frame():
    path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data'

    if not os.path.exists(join(path, 'biasframe_corrected')):
        os.mkdir(join(path, 'biasframe_corrected'))

    mask = np.load('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask_data_v2/final_mask/mask.npy')

    for i in range(16):
        data = np.load(join(path, 'biasframe', 'bf_'+str(i+1)+'.npy'))
        corrected_data = correct_using_mask(data, mask)

        np.save(join(path, 'biasframe_corrected', 'bf_'+str(i+1)+'.npy'), corrected_data)

def train_val_split():
    save_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/dataset_v4'

    src_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v4'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # d = ['train', 'test', 'val']
    # r = ['gt', 'x10', 'x20', 'x40']

    # for d_t in d:
    #     for r_t in r:
    #         if not os.path.exists(join(save_path, d_t, r_t)):
    #             os.makedirs(join(save_path, d_t, r_t))

    # val_list = np.random.choice(range(1, 75), 15)
    # test_list = np.random.choice(range(1, 75), 15)

    deprecated_list_path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/deprecated_data_list.txt'
    deprecated_list = []
    with open(deprecated_list_path) as f:
        l = f.readline()
        while l != '':
            deprecated_list.append(l[:-1])
            l = f.readline()
    
    data_list = ['scene_'+str(i) for i in range(1, 176)]
    data_list = [d for d in data_list if d not in deprecated_list]

    val_idx_list = np.random.choice(len(data_list), 20, replace=False)
    train_test_idx_list = [i for i in range(len(data_list)) if i not in val_idx_list]
    test_idx_list = np.random.choice(train_test_idx_list, 30, replace=False)

    val_idx_list.sort()
    test_idx_list.sort()

    val_list = [data_list[i] for i in val_idx_list]
    test_list = [data_list[i] for i in test_idx_list]
    train_list = [d for d in data_list if d not in val_list and d not in test_list]

    with open(join(save_path, 'train.txt'), 'w') as f:
        for tra in train_list:
            f.write(tra+'\n')

    with open(join(save_path, 'val.txt'), 'w') as f:
        for v in val_list:
            f.write(v+'\n')

    with open(join(save_path, 'test.txt'), 'w') as f:
        for t in test_list:
            f.write(t+'\n')


if __name__ == "__main__":
    # path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data'
    
    # if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask'):
    #     os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask')

    # if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_data'):
    #     os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_data')

    # if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_img'):
    #     os.mkdir('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_img')

    # if not os.path.exists('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask/mask.npy'):
    #     generate_mask()
    # correct_scene_data_v3()
    # train_val_split()
    # correct_bias_frame()
    generate_mask_v3()

    # for i in range(74):

    #     print('Processing scene %d...' % (i+1))

    #     if i == 5:
    #         expo = '500'
    #     else:
    #         expo = '800'

    #     scene_path = join(path, 'scene_'+str(i+1)+'_'+expo, 'unprocessed', 'frame.raw')
    #     # data = np.load(scene_path, allow_pickle=True)
    #     data = read_file(scene_path)
    #     mask, corrected_data = dpc_v2(data)

    #     np.save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/mask', 'mask_'+str(i+1)), mask)
    #     np.save(join('/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/corrected_data', 'data_'+str(i+1)), corrected_data)

