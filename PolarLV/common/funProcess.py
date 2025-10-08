import numpy as np



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def pairs_sequence(dcm_dir, roi_files) :
    if len(dcm_dir) != len(roi_files) :
        L_in_sqc = []
        for sqc in dcm_dir :
            nb_sqc = int(sqc[-3:]) - 4
            L_in_sqc.append(any(str(nb_sqc) in x[-8:] for x in roi_files))
        dcm_dir = [dcm_dir[i] for i in range(len(L_in_sqc)) if L_in_sqc[i] is True]
    return dcm_dir, roi_files

def norm_0_1 (array, offset=None) :
    if offset : return (array - offset[0])/(offset[1] - offset[0])
    else: return (array - np.min(array)) / (np.max(array) - np.min(array))

def getExp(value, base) :
    return np.log(value)/np.log(base)

