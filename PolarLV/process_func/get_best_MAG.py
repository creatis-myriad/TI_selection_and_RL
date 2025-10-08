import numpy as np
from scipy import stats
from scipy.stats import linregress
from PolarLV.common.funProcess import norm_0_1



def method_old_loss(dcm, remote_label, infarct_label, myocard_label, range_graph):
    L_U_value = []
    L_med_value = []
    L_cen_value = []
    for i in range (dcm.data.shape[-1]) :
        dcm_d = dcm.data[:,:,i]
        remote_data = dcm_d[remote_label]
        infarct_data = dcm_d[infarct_label]
        myocard_data = dcm_d[myocard_label]
        if np.all(myocard_data) == 0 : continue

        U1, p = stats.mannwhitneyu(infarct_data, remote_data)
        U2 = len(infarct_data)*len(remote_data) - U1
        U = min(U1,U2)
        D_med = abs(np.median(infarct_data) - np.median(remote_data)) / range_graph[1]
        D_center = ((np.median(infarct_data) + np.median(remote_data))/2)/ \
                ((range_graph[1]+range_graph[0])/2)

        L_U_value.append(U)
        L_med_value.append(D_med)
        L_cen_value.append(D_center)

    norm_U_value   = list(norm_0_1(L_U_value))
    idx_max = norm_U_value.index(max(norm_U_value))

    norm_med_value = list(1 - np.array(L_med_value))
    norm_cen_value = list(abs(1 - np.array(L_cen_value)))
    L_old_loss = list(np.sum([norm_U_value, norm_med_value, norm_cen_value], axis=0))

    return idx_max, L_old_loss

def method_std(dcm, remote_label, infarct_label, myocard_label, option="std"):
    L_U_value = []
    L_metric = []
    for i in range (dcm.data.shape[-1]) :
        dcm_d = dcm.data[:,:,i]
        remote_data = dcm_d[remote_label]
        infarct_data = dcm_d[infarct_label]
        myocard_data = dcm_d[myocard_label]
        if np.all(myocard_data) == 0 : continue

        U1, p = stats.mannwhitneyu(infarct_data, remote_data)
        U2 = len(infarct_data)*len(remote_data) - U1
        U = min(U1,U2)
        
        if option == "std" :
            val_metric  = np.std(myocard_data)
        elif option == "contrast":
            val_metric = abs(np.mean(infarct_data) - np.mean(remote_data))
        elif option == "cnr":
            val_metric = abs(np.mean(infarct_data) - np.mean(remote_data))/np.mean(remote_data)
        else:
            print("wrong option")
            exit()

        L_U_value.append(U)
        L_metric.append(val_metric)

    norm_U_value   = list(norm_0_1(L_U_value))
    idx_max = norm_U_value.index(max(norm_U_value))
    L_metric  = list(norm_0_1(max(L_metric) - L_metric))

    return idx_max, L_metric

def method_std_sat(dcm, remote_label, infarct_label, myocard_label, range_graph, lbd=0.75, option="std"):
    L_U_value = []
    L_metric = []
    L_val_remote_left = []
    L_val_remote_right = []
    for i in range (dcm.data.shape[-1]) :
        dcm_d = dcm.data[:,:,i]
        remote_data = dcm_d[remote_label]
        infarct_data = dcm_d[infarct_label]
        myocard_data = dcm_d[myocard_label]
        if np.all(myocard_data) == 0 : continue

        U1, p = stats.mannwhitneyu(infarct_data, remote_data)
        U2 = len(infarct_data)*len(remote_data) - U1
        U = min(U1,U2)

        if option == "std" :
            val_metric  = np.std(myocard_data)
        elif option == "contrast":
            val_metric = abs(np.mean(infarct_data) - np.mean(remote_data))
        elif option == "cnr":
            val_metric = abs(np.mean(infarct_data) - np.mean(remote_data))/np.mean(remote_data)
        else:
            print("wrong option")
            exit()

        val_remote_mean = np.mean(remote_data)
        val_remote_std  = np.std(remote_data)
        val_remote_left  = (val_remote_mean - 2*val_remote_std) - range_graph[0]
        val_remote_right = (val_remote_mean + 2*val_remote_std) - range_graph[1]

        if val_remote_left < 0: L_val_remote_left.append(1)
        else: L_val_remote_left.append(0)
        if val_remote_right > 0: L_val_remote_right.append(1)
        else: L_val_remote_right.append(0)
        L_U_value.append(U)
        L_metric.append(val_metric)

    norm_U_value   = list(norm_0_1(L_U_value))
    idx_max = norm_U_value.index(max(norm_U_value))
    norm_metric  = list(norm_0_1(max(L_metric) - L_metric))

    # Protection si jamais il n'y a que des 0 dans la liste
    if np.all(L_val_remote_left[idx_max:])  == 0 : L_val_remote_left[idx_max] = 1
    if np.all(L_val_remote_right[idx_max:]) == 0 : L_val_remote_right[-1] = 1

    idx_last_one_left = np.max(np.nonzero(L_val_remote_left)) + 1
    idx_first_one_right = np.min(np.nonzero(L_val_remote_right))
    slope, intercept, _, _, _ = linregress([idx_last_one_left, idx_first_one_right],[0,1])

    L_sat = list(np.concatenate((
        np.ones(idx_last_one_left), 
        slope*np.arange(0,idx_first_one_right-idx_last_one_left+1,1), 
        np.ones(len(L_val_remote_right)-idx_first_one_right-1),
    )))

    L_std_sat = list((1-lbd)*np.array(L_sat) + lbd*np.array(norm_metric))

    return idx_max, L_std_sat


def get_idx_best_MAG(dcm, roi, method="std+sat", range_graph_MAG=[2000,4000], option="std") :
    myocard_zone_MAG = roi.segments[0]['non-MI']
    remote_zone_MAG  = myocard_zone_MAG - roi.segments[0]['MI']
    infarct_zone_MAG = roi.segments[0]['MI']

    myocard_label_MAG = np.where(myocard_zone_MAG >= 0.1)
    remote_label_MAG  = np.where(remote_zone_MAG >= 0.1)
    infarct_label_MAG = np.where(infarct_zone_MAG >= 0.1)

    L_myocard_data_MAG = []
    for i in range (dcm.data.shape[-1]) :
        L_myocard_data_MAG.append(dcm.data[:,:,i][myocard_label_MAG])


    if method == "old_loss" :
        idx_max, L_idx_MAG = method_old_loss(
            dcm, 
            remote_label_MAG, 
            infarct_label_MAG, 
            myocard_label_MAG, 
            range_graph_MAG
        )
    elif method == "std" :
        idx_max, L_idx_MAG = method_std(
            dcm, 
            remote_label_MAG, 
            infarct_label_MAG, 
            myocard_label_MAG,
            option=option,
        )
    elif method == "std+sat" :
        idx_max, L_idx_MAG = method_std_sat(
            dcm, 
            remote_label_MAG, 
            infarct_label_MAG, 
            myocard_label_MAG, 
            range_graph_MAG, 
            lbd=0.75,
            option=option,
        )
    else :
        raise Exception("Wrong method. Try between : old_loss; std; metric+sat")

    idx_best_MAG = L_idx_MAG.index(min(L_idx_MAG[idx_max:]))
    return idx_best_MAG
