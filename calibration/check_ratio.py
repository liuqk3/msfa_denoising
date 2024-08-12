import os
import numpy as np
from os.path import join

# path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/dataset'
# print('Check test data...')
# for _, _, files in os.walk(join(path, 'test', 'gt')):
#     for file in files:
#         gt = np.load(join(path, 'test', 'gt', file))
#         x40 = np.load(join(path, 'test', 'x40', file))

#         print(file)
#         print(np.mean(gt))
#         print(np.mean(x40)*40)
#         print('\n')

# print('Check train data...')
# for _, _, files in os.walk(join(path, 'train', 'gt')):
#     for file in files:
#         gt = np.load(join(path, 'train', 'gt', file))
#         x40 = np.load(join(path, 'train', 'x40', file))

#         print(file)
#         print(np.mean(gt))
#         print(np.mean(x40)*40)
#         print('\n')

# print('Check val data...')
# for _, _, files in os.walk(join(path, 'val', 'gt')):
#     for file in files:
#         gt = np.load(join(path, 'val', 'gt', file))
#         x40 = np.load(join(path, 'val', 'x40', file))

#         print(file)
#         print(np.mean(gt))
#         print(np.mean(x40)*40)
#         print('\n')

path = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v2'
print('Checking data...')

# dir_list = os.listdir(path)
# for dir in dir_list:
for i in range(1, 67):
    dir = 'scene_'+str(i)
    scene_path = join(path, dir)
    scene_800 = np.load(join(scene_path, '800.npy'))
    scene_80 = np.load(join(scene_path, '80.npy'))
    scene_40 = np.load(join(scene_path, '40.npy'))
    scene_20 = np.load(join(scene_path, '20.npy'))

    if np.min(scene_800) < 0:
        print(np.min(scene_800))
    if np.min(scene_80) < 0:
        print(np.min(scene_80))
    if np.min(scene_40) < 0:
        print(np.min(scene_40))
    if np.min(scene_20) < 0:
        print(np.min(scene_20))

    print(dir)
    # print('800: '+str(np.mean(scene_800)))
    # print('80: '+str(np.mean(scene_80)*10))    
    # print('40: '+str(np.mean(scene_40)*20))
    # print('20: '+str(np.mean(scene_20)*40))

    print('\n')