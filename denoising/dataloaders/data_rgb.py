import os
from .dataset_rgb import DataLoaderTrain, DataLoaderTrainNoise, DataLoaderVal, DataLoaderTrainSyn, DataLoaderVal_v2
from .dataset_rgb_decouple import DataLoaderTrainSynCB, DataLoaderTrainSynCBParam, DataLoaderValCBParam, DataLoaderTrainSynColIndex, DataLoaderValColIndex
from .dataset_rgb_decouple import DataLoaderTrainRealCoord
from .dataset_rgb_decouple import DataLoaderVal_all_cb
from .dataset_rgb_decouple import DataLoaderTrainRealColIndex
from .dataset_rgb_decouple import DataLoaderTrainSynCoord

def get_training_data(rgb_dir, list_path, ratio, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, list_path, ratio, img_options, None)
    # return DataLoaderTrainNoise(rgb_dir, ratio, img_options, None)

def get_validation_data(rgb_dir, list_path, ratio):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, list_path, ratio, None)

def get_validation_data_cb_param(rgb_dir, list_path, ratio):
    assert os.path.exists(rgb_dir)
    # return DataLoaderValCBParam(rgb_dir, list_path, ratio, None)
    return DataLoaderValColIndex(rgb_dir, list_path, ratio, None)

def get_validation_data_all_cb(rgb_dir, list_path, ratio):
    assert os.path.exists(rgb_dir)
    # return DataLoaderValCBParam(rgb_dir, list_path, ratio, None)
    return DataLoaderVal_all_cb(rgb_dir, list_path, ratio, None)


def get_test_data(rgb_dir, list_path):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)

def get_training_syn_data(rgb_dir, list_path, ratio, noisemodel, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrainSyn(rgb_dir, list_path, ratio, noisemodel, img_options, None)
    # return DataLoaderTrainNoise(rgb_dir, ratio, img_options, None)

# def get_test_data_SR(rgb_dir):
#     assert os.path.exists(rgb_dir)
#     return DataLoaderTestSR(rgb_dir, None)


def get_training_syn_data_cb(rgb_dir, list_path, ratio, noisemodel, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrainSynCB(rgb_dir, list_path, ratio, noisemodel, img_options, None)

def get_training_real_data_cb_param(rgb_dir, list_path, ratio, img_options=None):
    assert os.path.exists(rgb_dir)
    # return DataLoaderTrainSynCBParam(rgb_dir, list_path, ratio, noisemodel, img_options, None)
    return DataLoaderTrainRealColIndex(rgb_dir, list_path, ratio, img_options=img_options)

def get_training_syn_data_cb_param(rgb_dir, list_path, ratio, noisemodel, img_options=None):
    assert os.path.exists(rgb_dir)
    # return DataLoaderTrainSynCBParam(rgb_dir, list_path, ratio, noisemodel, img_options, None)
    return DataLoaderTrainSynColIndex(rgb_dir, list_path, ratio, noisemodel, img_options, None)

def get_training_syn_data_coord(rgb_dir, list_path, ratio, noisemodel, img_options=None):
    assert os.path.exists(rgb_dir)
    # return DataLoaderTrainSynCBParam(rgb_dir, list_path, ratio, noisemodel, img_options, None)
    return DataLoaderTrainSynCoord(rgb_dir, list_path, ratio, noisemodel, img_options, None)

def get_training_real_data_coord(rgb_dir, list_path, ratio, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrainRealCoord(rgb_dir, list_path, ratio, img_options, None)