import numpy as np
import sys

sys.path.append('../../') 

from nn_libr.math import interp



def Get_WeightsVertices(radial, theta, shape=(512,16), method="P2C") :
    X,Y = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 2*np.pi, shape[0]))
    cartesian_coords = np.array([X.ravel(), Y.ravel()]).T

    coords_data = np.where(~np.isnan(radial))
    r = radial[coords_data]
    t = theta[coords_data] + np.pi
    t[t>2*np.pi] -= 2*np.pi

    # Extended coordinates for continuity (10 degrees)
    idx_extended_nega = np.where(t>2*np.pi-0.175)
    idx_extended_posi = np.where(t<0.175)
    coords_extended_nega = t[idx_extended_nega] - 2*np.pi
    coords_extended_posi = t[idx_extended_posi] + 2*np.pi

    r = np.concatenate([r, r[idx_extended_nega], r[idx_extended_posi]])
    t = np.concatenate([t, coords_extended_nega, coords_extended_posi])
    polar_coords = np.array([r, t]).T

    if method == "P2C" :
        vertices, weights = interp.interp_weights(polar_coords, cartesian_coords)
    else :
        vertices, weights = interp.interp_weights(cartesian_coords, polar_coords)
    return vertices, weights, idx_extended_nega, idx_extended_posi


def Polar2Cartesian(vertices, weights, values, convX=9, shape=(512,16), fill_value=np.nan):
    interp_data = interp.interpolate_known_indices(vertices, weights, values, fill_value=fill_value)
    tmp_interp_data = np.reshape(interp_data, shape)

    tmp_start = tmp_interp_data[:convX-1]
    tmp_end   = tmp_interp_data[-convX+1:]
    extended_interp_data = np.concatenate([tmp_end, tmp_interp_data, tmp_start])
    return extended_interp_data

def Cartesian2Polar(vertices, weights, values, fill_value=np.nan):
    return interp.interpolate_known_indices(vertices, weights, values.ravel(), fill_value=fill_value)




