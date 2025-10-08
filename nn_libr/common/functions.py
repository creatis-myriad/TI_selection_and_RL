import numpy as np



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def ComputeTransmurality(infarct_zone, theta, radial) :
    coords_data = np.where(~np.isnan(theta))
    t = theta[coords_data]
    r = radial[coords_data]
    L_trans = []
    L_coeff = []
    start_t = 0
    for interval_t in np.linspace(0, 2*np.pi, 37)[1:]:
        idx_interval_t = np.where((t>=start_t) & (t<interval_t))
        transmurality = infarct_zone[coords_data][idx_interval_t]

        if len(transmurality) != 0 :
            if np.mean(transmurality)>0 :
                L_coeff_tmp =  []
                radcoord_inter_t = r[idx_interval_t]
                start_r = 0
                for interval_r in np.linspace(0, 1, 11)[1:]:
                    idx_interval_r = np.where((radcoord_inter_t>=start_r) & (radcoord_inter_t<interval_r))
                    if len(idx_interval_r[0]) == 0: continue
                    L_coeff_tmp.append(np.mean(transmurality[idx_interval_r]))
                    start_r = np.copy(interval_r)

                L_coeff.append(np.mean(L_coeff_tmp))
                L_trans.append(np.mean(transmurality))
        start_t = np.copy(interval_t)

    L_coeff = np.array(L_coeff)
    L_trans = np.array(L_trans)
    return np.sum(L_coeff*L_trans)/np.sum(L_coeff)

def ComputeEndoSurfArea(myocard, radial) :
    coords_data = np.where(~np.isnan(radial))
    r = radial[coords_data]
    idx_interval_r = np.where(r<0.25)
    esa = myocard[coords_data][idx_interval_r]
    return np.mean(esa)

def ComputeInfarctExtent(myocard):
    return np.nansum(myocard)/myocard.size

def CenterOfMass(input, weights=None) :
    idx_CoM_x, idx_CoM_y = np.average(input, axis=0, weights=weights)
    return np.array([idx_CoM_x, idx_CoM_y], dtype=int)

def ComputeAngleJunction(myocard, coords_data, point, center):
    x_coord, y_coord  = coords_data
    grid = np.stack([x_coord, y_coord, myocard], axis=-1)
    idx_CoM_x, idx_CoM_y = CenterOfMass(grid[:,:-1], weights=grid[:,-1])

    com = [idx_CoM_x, idx_CoM_y]
    v1 = point - center
    v2 = com - center
    theta_rad = np.arctan2(v1[1],v1[0]) - np.arctan2(v2[1],v2[0])
    if theta_rad > np.pi  : theta_rad -= 2*np.pi
    if theta_rad < -np.pi : theta_rad += 2*np.pi
    theta_deg = -(theta_rad*180)/np.pi
    return theta_deg, com


