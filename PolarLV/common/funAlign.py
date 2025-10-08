#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:12:31 2021

@author: zheng
"""

import sys, os
sys.path.append('../../') 

import numpy as np
import warnings
import time
import scipy.interpolate as interpolate
import PolarLV.math.interp as interp

from copy import deepcopy
import matplotlib.pyplot as plt
import PolarLV.common.funPlot as funPlot
import PolarLV.common.funCoord as funCoord

def alignment(dcm, roi, roiRef, 
              idx_slice_ref = None,
              mtd='linear', fill_value=0, 
              removeMissingAngles = True, 
              dcmReplaceNonMI = False, 
              segmentsExtrapo = {'non-MI': 'copy', 'MI': 0, 'NR': 0}):
    """ Alignment 
        parameters:
            dcm: formated MRT data object
            roi: formated segmentation object
            roiRef: reference object
            mtd: string, 'linear', 'nearest', 'cubic',
                use 
                'linear': optimised linear interpolation (fastest) by avoid recomputing of indices of the vertices
                'nearest' and 'cubic' are method from interpolate.griddata
            fill_value: filled value for region out of interest
            removeMissingAngles: boolean, if 1 the open angle (caused by myo opening) 
                is removed from normalized data
            dcmReplaceNonMI: if True rescaled dcm data will replace 'non-MI' segmentation in aligned segments (seg_interp)
            segmentsExtrapo: extrapolation technique for ROI of missing apex/base slices 
                coordinate of missing apex/base slices will be copy of nearest slices (need to be improved)
                'copy': segmentations will be a copy of nearest slice
                0: no infarct/MVO
                1: full of infarct/MVO
        Returns:
            data_interp, seg_interp, missingAngles_ref
    """
    # numSlices = dcm.dataResampled.shape[-1]
    t_start = time.time()
    coordinates_tmp = roi.coordinates.copy()
    segmentsResampled_tmp = deepcopy(roi.segmentsResampled)
    dataResampled_tmp = dcm.dataResampled.copy()   
    
    #%% deal with segments missing towards apex or base 
    # currently a copy of the coordinates (radial, circumferential) of the nearest slice are assigned to the apex/base
    # the image content was copied, the segmentations are assinged 0 values
    # however if no infarct present at nearest silces, the assinged value will be 0
    
    slice_with_coord = np.where([~np.all(np.isnan(roi.coordinates[:,:,-1,i])) \
                                  for i in range(roi.coordinates.shape[-1])])[0]
    sliceLowest = slice_with_coord[0]; sliceHighest = slice_with_coord[-1] # the idx of slices with segments closest to the apex and base
    if sliceLowest > roi.idxes['apex']:
        coords_apex = coordinates_tmp[:,:,:,sliceLowest].copy()
        coords_apex[~np.isnan(coords_apex[:,:,-1]),-1] = 0 # slice apex z=0 
        coordinates_tmp = np.append(coords_apex[:,:,:,np.newaxis], coordinates_tmp, axis=-1)
        segments_apex = deepcopy(segmentsResampled_tmp[sliceLowest])
        for key in segments_apex:
            if segmentsExtrapo[key] != 'copy':
                if np.nanmax(segments_apex[key]) > 0:
                    segments_apex[key][~np.isnan(segments_apex[key])] = segmentsExtrapo[key]
                else:
                    segments_apex[key][~np.isnan(segments_apex[key])] = 0
        segmentsResampled_tmp.insert(0, segments_apex)
        dataResampled_tmp = np.append(dataResampled_tmp[:,:,sliceLowest:sliceLowest+1].copy(), 
                                      dataResampled_tmp, axis=-1) 
    if sliceHighest < roi.idxes['valve']:
        coords_valve = coordinates_tmp[:,:,:,sliceHighest].copy()
        coords_valve[~np.isnan(coords_valve[:,:,-1]),-1] = 1 # slice apex z=1
        coordinates_tmp = np.append(coordinates_tmp, coords_valve[:,:,:,np.newaxis], axis=-1)
        segments_valve = deepcopy(segmentsResampled_tmp[sliceHighest])
        for key in segments_valve:
            if segmentsExtrapo[key] != 'copy':
                if np.nanmax(segments_valve[key]) > 0:
                    segments_valve[key][~np.isnan(segments_valve[key])] = segmentsExtrapo[key]
                else:
                    segments_valve[key][~np.isnan(segments_valve[key])] = 0
        segmentsResampled_tmp.append(segments_valve)
        dataResampled_tmp = np.append(dataResampled_tmp,
                                      dataResampled_tmp[:,:,sliceHighest:sliceHighest+1].copy(), 
                                      axis=-1) 
    print('\ndealing with missing apex/base spends {:.3f} s.\n'.format(time.time()-t_start))
  
    #%%
    numSlices = len(segmentsResampled_tmp)
    
    if idx_slice_ref :
        # The values of the myocard for the z axis must be change (between 0 and 1)
        copy_coordsRef = roiRef.coordinates.copy()
        for i in range (roiRef.coordinates.shape[-1]) :
            copy_coordsRef[:,:,:,i] = roiRef.coordinates[:,:,:,idx_slice_ref]
            copy_coordsRef[:,:,2,i] = (copy_coordsRef[:,:,2,i]/0.5)*(i/(roiRef.coordinates.shape[-1]-1))

        coordsRef_ravel = funCoord.ravel_coordinates(copy_coordsRef)

    else :
        coordsRef_ravel = roiRef.ravel_coordinates()

    coords_ravel =  funCoord.ravel_coordinates(coordinates_tmp)


    idx_notnan = np.where(~(np.isnan(coords_ravel).any(axis=1)))[0]
    idx_notnan_ref = np.where(~np.isnan(coordsRef_ravel).any(axis=1))[0]
    
    # initialize interpolated data and segments
    data_interp = np.full(coordsRef_ravel.shape[0], fill_value=np.nan) # raveled data
    seg_interp = [{key: np.full(roiRef.coordinates.shape[0:2], fill_value=fill_value, dtype=float) \
                   for key in roi.segmentsResampled[0].keys()} \
                  for i in range(roiRef.coordinates.shape[-1])]
        
    coords_notnan = coords_ravel[idx_notnan, :] # coordinates known [:,3] ndarray
    coords_notnan_interp = coordsRef_ravel[idx_notnan_ref, :] # coordinates to interp
    
    ## take care of the continuity of data around 0 degree from origine 
    #  by adding data with radial > 2*pi and < 0
    idx_add_nega = np.where(coords_ravel[:,1] > 2*np.pi - 0.35)[0] # around 15 degree
    idx_add_posi = np.where(coords_ravel[:,1] < 0.35)[0]
    
    coords_add_nega = coords_ravel[idx_add_nega, :]
    coords_add_nega[:,1] = coords_add_nega[:,1] - 2*np.pi
    coords_add_posi = coords_ravel[idx_add_posi, :]
    coords_add_posi[:,1] = coords_add_posi[:,1] + 2*np.pi

    coords_notnan_extend = np.concatenate((coords_notnan, coords_add_nega, coords_add_posi), axis=0)
    idx_notnan_extend = np.concatenate((idx_notnan, idx_add_nega, idx_add_posi))
        
    ## ------ solution 1 (Delaunay triangulation based, super slow...)    
    if mtd in ['nearest', 'cubic']:
        # coords = (rad[idx_notnan], ang[idx_notnan], z[idx_notnan])
        # coords_interp = (rad_ref[idx_notnan_ref], ang_ref[idx_notnan_ref], z_ref[idx_notnan_ref])
        
        data_interp_notnan = interpolate.griddata(tuple(coords_notnan_extend.T), # tuple(coords_notnan.T), 
                                                  dataResampled_tmp.ravel()[idx_notnan_extend], # dcm.dataResampled.ravel()[idx_notnan], 
                                                  tuple(coords_notnan_interp.T), 
                                                  method=mtd, 
                                                  fill_value=fill_value) 
        data_interp[idx_notnan_ref] = data_interp_notnan
        data_interp = np.reshape(data_interp, roiRef.coordinates[:,:,0,:].shape)
        
        for key in roi.segmentsResampled[0].keys():
            if key != 'non-MI':
                segs = np.concatenate([segmentsResampled_tmp[i][key][:,:,np.newaxis] \
                            for i in range(numSlices)], axis=-1)
                seg_interp_notnan =  interpolate.griddata(tuple(coords_notnan_extend.T), 
                                                          segs.ravel()[idx_notnan_extend], 
                                                          tuple(coords_notnan_interp.T), 
                                                          method=mtd, 
                                                          fill_value=fill_value)
                seg_interp_tmp = np.full(coordsRef_ravel.shape[0], fill_value=np.nan)
                seg_interp_tmp[idx_notnan_ref] = seg_interp_notnan
                seg_interp_tmp = np.reshape(seg_interp_tmp, roiRef.coordinates[:,:,0,:].shape)  
                for i in range(roiRef.coordinates.shape[-1]):
                    seg_interp[i][key] = seg_interp_tmp[:,:,i] 
                

    ## ------ solution 2 (super slow...)
    # f_interp = interpolate.LinearNDInterpolator((rad[idx_notnan], ang[idx_notnan], z[idx_notnan]), 
    #                                             data[idx_notnan], fill_value=0)
    # data_iterp_notnan_1 = f_interp((rad_ref[idx_notnan_ref], 
    #                                 ang_ref[idx_notnan_ref], 
    #                                 z_ref[idx_notnan_ref]))
    
    ## ------ solution 3 use rbf (radial basis function based, ca)
    # fun_rbf_data = interpolate.Rbf(rad[idx_notnan], ang[idx_notnan], z[idx_notnan], 
    #                                dcm.dataResampled.ravel()[idx_notnan], function='linear')
    # data_interp_notnan = fun_rbf_data(rad_ref[idx_notnan_ref], 
    #                                   ang_ref[idx_notnan_ref], 
    #                                   z_ref[idx_notnan_ref])
    
    ## ------ solution 4 http://rncarpio.github.io/delaunay_linterp/ waiting for test
    
    ## ------ solution 5 
    # store the indices of the vertices and the weights for new grid values
    if mtd == 'linear':
        vertices, weights = interp.interp_weights(coords_notnan_extend, coords_notnan_interp)
        data_interp_notnan = interp.interpolate_known_indices(vertices, weights, 
                                                              dataResampled_tmp.ravel()[idx_notnan_extend], 
                                                              fill_value=0)
        data_interp[idx_notnan_ref] = data_interp_notnan
        data_interp = np.reshape(data_interp, roiRef.coordinates[:,:,0,:].shape)
    
        for key in roi.segmentsResampled[0].keys():
            if key != 'non-MI':
                segs = np.concatenate([segmentsResampled_tmp[i][key][:,:,np.newaxis] \
                           for i in range(numSlices)], axis=-1)
                seg_interp_notnan = interp.interpolate_known_indices(vertices, weights, 
                                                                     segs.ravel()[idx_notnan_extend], 
                                                                     fill_value=fill_value)
                seg_interp_tmp = np.full(coordsRef_ravel.shape[0], fill_value=np.nan)
                seg_interp_tmp[idx_notnan_ref] = seg_interp_notnan
                seg_interp_tmp = np.reshape(seg_interp_tmp, roiRef.coordinates[:,:,0,:].shape)  
                for i in range(roiRef.coordinates.shape[-1]):
                    seg_interp[i][key] = seg_interp_tmp[:,:,i] 
                    
    # deal with 'non-MI'                 
    for i in range(roiRef.coordinates.shape[-1]): 
        seg_interp[i]['non-MI'] *= np.nan # convert all data to None
        seg_interp[i]['non-MI'][roiRef.segmentsResampled[i]['non-MI']>0] = 1 
    if dcmReplaceNonMI:
        data_interp_rescaled = (data_interp - dcm.dataResampled.min())/(dcm.dataResampled.max()-dcm.dataResampled.min())
        for i in range(roiRef.coordinates.shape[-1]):
            seg_interp[i]['non-MI'] = data_interp_rescaled[:,:,i]

    # deal with missing angles at myoOpening    
    missingAngles_ref, idx_missStart_ref = interp_missing_angles(roi, roiRef) # interpolate missingAngles
    missingAngles_ref[missingAngles_ref>=2*np.pi] -= 2*np.pi # solve myoOpening ends > 2*pi
    if removeMissingAngles:
        # remove values from missing angles in standardized data
        if not np.isnan(idx_missStart_ref):
            for i in range(idx_missStart_ref, data_interp.shape[-1]):
                idx = np.logical_and(roiRef.coordinates[:,:,1,i] < missingAngles_ref[1,i],
                                     roiRef.coordinates[:,:,1,i] > missingAngles_ref[0,i])
                data_interp[idx,i] = np.nan
                for key, val in seg_interp[i].items():
                    if len(val) > 0:
                        seg_interp[i][key][idx] = np.nan
    print('\nAlign data and segments takes {:.3f} s.\n'.format(time.time()-t_start))
    return data_interp, seg_interp, missingAngles_ref


def interp_missing_angles(roi, roiRef):
    """ compute (interpolate) missing angles in the reference axes 
    """
    missingAngles = roi.compute_missingAngles() # get missing angles caused by myoOpening
    missingAngles_ref = roiRef.compute_missingAngles()
    if np.all(np.isnan(missingAngles)): # no missing angles
        idx_missStart_ref = np.nan
    else: 
        idx_missAng= np.where(~np.isnan(missingAngles[0,:]))[0]
        z_vals = roi.get_z_vals()
        z_vals_ref = roiRef.get_z_vals()
        idx_missStart_ref = np.where(z_vals_ref <= z_vals[idx_missAng[0]])[0][-1]
        # idx_missStart_ref = np.where(z_vals_ref <= z_vals[idx_missAng[0]-1])[0][0] # problem if z_vals[idx_missAng[0]-1] = nan
        angleMyoClosing = missingAngles[:, idx_missAng[0]].mean() # artifacial myoClosing
        for i in range(2): # interpolation for missing angles in standardized data
            f_interp = interpolate.interp1d(np.array(z_vals)[np.append(idx_missAng[0]-1,idx_missAng)], 
                                            np.append(angleMyoClosing, missingAngles[i, idx_missAng]),
                                            kind='linear', fill_value='extrapolate')
            missingAngles_ref[i, idx_missStart_ref:] = f_interp(z_vals_ref[idx_missStart_ref:])
    return missingAngles_ref, idx_missStart_ref