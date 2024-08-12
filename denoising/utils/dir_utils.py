import os
from natsort import natsorted
from glob import glob

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_last_path(path, session):
	x = natsorted(glob(os.path.join(path,'*%s'%session)))[-1]
	return x

def get_data_list(path):
    data_list = []
    with open(path) as f:
        l = f.readline()
        while l != '':
            data_list.append(l[:-1])
            l = f.readline()

    return data_list