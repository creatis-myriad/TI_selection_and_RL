#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 17:34:18 2021

@author: zheng

Fast linear 3d interpolation
Economize the time of recalculating the indices of the vertices of the enclosing simplex 
and the weights which might be recomputed several times

Modified from:
https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
"""
import sys
sys.path.append('../../')

import scipy.spatial.qhull as qhull
import numpy as np
from PolarLV.math.improvedDelauney import ImprovedTessellation

def interp_weights(xyz, uvw, extrapolation=False):
    d = xyz.shape[-1]
    if extrapolation:
        # this solution has problem super super slow ...
        tri = ImprovedTessellation(xyz)
        simplex= tri.find_simplex(uvw)
    else:
        tri = qhull.Delaunay(xyz)
        simplex = tri.find_simplex(uvw)        
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    weights[simplex == -1] = np.nan # assign nan weights to extrapolation points
    return vertices, weights

def interpolate_known_indices(vertices, weights, vals, fill_value=np.nan):
    vals_interp = np.einsum('nj,nj->n', np.take(vals, vertices), weights)
    vals_interp[np.any(np.isnan(weights), axis=1)] = fill_value
    return vals_interp

def linear_3d_interpolate(xyz, vals, uvw, fill_value=np.nan):
    vertices, weights = interp_weights(xyz, uvw)
    vals_interp = interpolate_known_indices(vertices, weights, vals, fill_value)
    return vals_interp
