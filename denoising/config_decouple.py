
from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):

    def __init__(self, config_yaml: str=None, config_override: List[Any] = []):

        self._C = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = False

        self._C.MODEL = CN()
        self._C.MODEL.MODE = 'global'
        self._C.MODEL.RATIO = 'mix'
        self._C.MODEL.SESSION = 'ps128_bs1'

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 1
        self._C.OPTIM.NUM_EPOCHS = 100
        self._C.OPTIM.LR_INITIAL = 1e-4
        self._C.OPTIM.BETA1 = 0.5

        self._C.DATA = 'syn'
        self._C.NOISEMODEL = 'PgRCBU'

        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 3
        self._C.TRAINING.RESUME = False
        self._C.TRAINING.SAVE_IMAGES = False

        self._C.TRAINING.DATA_DIR = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v4'
        self._C.TRAINING.TRAIN_FILE = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/dataset_v4/train.txt'
        self._C.TRAINING.VAL_FILE = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/dataset_v4/testval.txt'
        if self._C.DATA == 'syn':
            self._C.TRAINING.SAVE_DIR = 'noise_decouple_ckpt/unet_decoupled_posemb_plus_25c_6'
        else:
            self._C.TRAINING.SAVE_DIR = 'ckpt_density_guided_noisemodel/real_data_4'
        self._C.TRAINING.TRAIN_PS = 128
        self._C.TRAINING.VAL_PS = 128

        if config_yaml is not None:
            self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        self._C.freeze()

    def dump(self, file_path: str):
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()
