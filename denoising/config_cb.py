#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:35:48 2019

@author: aditya
"""

r"""This module provides package-wide configuration management."""
from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overriden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Extended Summary
    ----------------
    This class definition contains default values corresponding to ``joint_training`` phase, as it
    is the final training phase and uses almost all the configuration parameters. Modification of
    any parameter after instantiating this class is not possible, so you must override required
    parameter values in either through ``config_yaml`` file or ``config_override`` list.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        ALPHA: 1000.0
        BETA: 0.5

    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048, "BETA", 0.7])
    >>> _C.ALPHA  # default: 100.0
    1000.0
    >>> _C.BATCH_SIZE  # default: 256
    2048
    >>> _C.BETA  # default: 0.1
    0.7

    Attributes
    ----------
    """

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
        self._C.OPTIM.NUM_EPOCHS = 250
        # self._C.OPTIM.NEPOCH_DECAY = [100]
        self._C.OPTIM.LR_INITIAL = 1e-4
        self._C.OPTIM.BETA1 = 0.5

        self._C.DATA = 'syn'
        self._C.NOISEMODEL = 'PgRCBU'

        # self._C.DATA = 'real'
        # self._C.NOISEMODEL = ''

        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 3
        self._C.TRAINING.RESUME = False
        self._C.TRAINING.SAVE_IMAGES = False
        # self._C.TRAINING.TRAIN_DIR = 'images_dir/train'
        # self._C.TRAINING.TRAIN_DIR = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/dataset_v3/'
        # self._C.TRAINING.VAL_DIR = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/dataset_v3/'
        self._C.TRAINING.DATA_DIR = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/scene_data_v4'
        self._C.TRAINING.TRAIN_FILE = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/dataset_v4/train.txt'
        self._C.TRAINING.VAL_FILE = '/home/jiangyuqi/Desktop/HSI_denoising_demosaicing/data/dataset_v4/testval.txt'
        # self._C.TRAINING.VAL_DIR = 'images_dir/val'
        # self._C.TRAINING.SAVE_DIR = 'checkpoints_real'
        if self._C.DATA == 'syn':
            # self._C.TRAINING.SAVE_DIR = 'ckpt_trans_competing_network_syn/'+self._C.DATA+str(self._C.MODEL.RATIO)+'_'+self._C.NOISEMODEL
            # self._C.TRAINING.SAVE_DIR = 'ckpt_density_guided_ablation/DGNet_6'
            self._C.TRAINING.SAVE_DIR = 'ffc_decouple/unet_cb_concat_colindex_v3_with_mapping'
            # self._C.TRAINING.SAVE_DIR = 'checkpoints/motivation/B'
        else:
            # self._C.TRAINING.SAVE_DIR = 'ckpt_trans_competing_network/Pan'
            self._C.TRAINING.SAVE_DIR = 'ckpt_density_guided_noisemodel/real_data_4'
        self._C.TRAINING.TRAIN_PS = 128
        self._C.TRAINING.VAL_PS = 128

        # Override parameter values from YAML file first, then from override list.
        if config_yaml is not None:
            self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()
