import os
import pickle
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append('../../') 

from nn_libr.common.functions import rgb2gray


def Load_dataset(L_path, L_pat_name, idx_split_pat, input_type=None, input_seg=False) :
    print("\nLoading", input_type, "input data ...")
    dataset = []
    L_name_pat = []
    L_idx_split = []

    idx_split = 0
    for path in L_path :
        if "pre" in input_type and "post" in path : continue
        if "post" in input_type and "pre" in path : continue

        if L_pat_name[idx_split_pat]+"/" in path : 
            L_idx_split.append(idx_split)

        if input_type == "BE" :
            if "/"+L_pat_name[idx_split_pat] in path : 
                L_idx_split.append(idx_split)
            data = tf.keras.utils.load_img(path)
            data = rgb2gray(np.array(data))/255.0

        else :
            if not input_seg :
                with open(path, 'rb') as filedcm:
                    dcm = pickle.load(filedcm) # load the images
                data = dcm.data
            else:
                with open(path, 'rb') as fileroi:
                    roi = pickle.load(fileroi) # load the images
                data = roi.segments[0]['MI']
                data = np.expand_dims(data, axis=-1)

        if "BE"  in input_type : name=path.split("/")[-1].split(".")[0]
        if "LGE" in input_type : name=path.split("/")[-2:-1]
        if "T1"  in input_type : name=path.split("/")[-3:-1]
        if "MAG" in input_type : name=path.split("/")[-4:-1]

        L_name_pat.append(os.path.join(*name))
        dataset.append(data)
        idx_split+=1

    return np.array(dataset), L_name_pat, L_idx_split

