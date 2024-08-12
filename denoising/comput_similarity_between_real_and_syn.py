import numpy as np
import os
from utils.image_utils import compute_kl_and_ce
import json
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


with open('data/dataset_v4/testval.txt', 'r') as f:
    scene_list = f.readlines()
    scene_list = [s.replace('\n', '') for s in scene_list]
# scene_list.append('none')
# import pdb; pdb.set_trace()
ratios = [10, 20, 40]
img_dir = 'img_noisemodel_raw'

noise_models_dict = {
                'ELD': 'PgRCbU',
                'HSI-Real': 'PgC',
                'Homo': 'g',
                'Hetero': 'pg',
                'Ours': 'PgRCBU',
                'DANet': 'DANet',
                'Complex': 'complex'
            }
noise_models_dict_inverse = {v: k for k, v in noise_models_dict.items()}
noise_models = list(noise_models_dict.values())


metrics = {}

for ratio in ratios:
    if ratio not in metrics:
        metrics[ratio] = {
            'kl': {},
            'ce': {},
            'psnr': {},
            'ssim': {}
        }

    for idx, scene in enumerate(scene_list):
        # if idx >= 3:
        #     break

        print('ratio: {}, {}/{}'.format(ratio, idx+1, len(scene_list)))
        
        real_img_path = os.path.join(img_dir, 'real_data', str(ratio), '{}.npy'.format(idx))
        real_img = np.load(real_img_path)
        real_img = np.clip(real_img, 0, 1)
        # real_img = (real_img * 255).astype(np.uint8).astype(np.float32) / 255

        clean_img_path = os.path.join(img_dir, 'gt', str(ratio), '{}.npy'.format(idx))
        clean_img = np.load(clean_img_path)
        clean_img = np.clip(clean_img, 0, 1)
        # clean_img = (clean_img * 255).astype(np.uint8).astype(np.float32) / 255

        for nm in noise_models:
            # nm = 'PgRCBU' #TODO:
            for metric_name in metrics[ratio].keys():
                if noise_models_dict_inverse[nm] not in metrics[ratio][metric_name]:
                    metrics[ratio][metric_name][noise_models_dict_inverse[nm]] = []

            syn_img_path = os.path.join(img_dir, nm, str(ratio), '{}_noisy.npy'.format(idx))
            syn_img = np.load(syn_img_path)
            syn_img = np.clip(syn_img, 0, 1)
            # syn_img = (syn_img * 255).astype(np.uint8).astype(np.float32) / 255

            kl, ce = compute_kl_and_ce(real_img, syn_img, clean_img, channel_wise=False)
            psnr = peak_signal_noise_ratio(np.transpose(real_img, (1,2,0)), np.transpose(syn_img, (1,2,0)))
            ssim = structural_similarity(np.transpose(real_img, (1,2,0)), np.transpose(syn_img, (1,2,0)), multichannel=True)

            metrics[ratio]['ce'][noise_models_dict_inverse[nm]].append(ce)
            metrics[ratio]['kl'][noise_models_dict_inverse[nm]].append(kl)
            metrics[ratio]['psnr'][noise_models_dict_inverse[nm]].append(psnr)
            metrics[ratio]['ssim'][noise_models_dict_inverse[nm]].append(ssim)


for ratio in metrics.keys():
    for metric_name, res_d in metrics[ratio].items():
        for noise_model_name, res_list in res_d.items():
            metrics[ratio][metric_name][noise_model_name] = float(np.mean(res_list))
print(metrics)
save_path = os.path.join(img_dir, 'similarity.json')
json.dump(metrics, open(save_path, 'w'), indent=4)
print('save to {}'.format(save_path))

        
