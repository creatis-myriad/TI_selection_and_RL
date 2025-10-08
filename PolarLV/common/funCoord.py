#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:41:28 2020

@author: zheng
"""

import sys, os
sys.path.append('../../') 

import numpy as np
import warnings
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

import PolarLV.common.funImg as funImg
import PolarLV.common.funArray as funArray
import PolarLV.common.funPlot as funPlot


def compute_polar_coordinates(dcm, roi, flag_plot=False, figurePause=3):
    """ Compute polar coordinates of region of interest (roi) 
            which is the left ventrical myocardium
        Paramters: 
            dcm: instance of object funObj.DicomObj
            roi: instance of object funObj.RoiObj
        Returns:
            coordinates: [N_H, N_W, 3, numSlices], includes
                radial: [:,:,0,:] 
                angle:  [:,:,1,:] 
                Z:      [:,:,2,:] 
    """
    numSlices = len(roi.segmentsResampled)
    n_h, n_w = roi.segmentsResampled[0]['non-MI'].shape
    coordinates = np.full((n_h, n_w, 3, numSlices), fill_value=np.nan)
    # Z
    Z = compute_z(dcm.USliceLocation, dcm.spacingResampled[-1], 
                  roi.idxes['apex'], roi.idxes['valve'])
    
    for i in range(numSlices):
        if i < np.floor(roi.idxes['apex']) or i > np.ceil(roi.idxes['valve']):
            continue # don't care if the slice is out of apex-valve range
        # get region of no interest (roni) the region covers the angle of myoOpening - origine
        roni = funImg.find_roni(roi.endoCentersResampled[:,i], 
                                roi.myoOpeningsResampled[:,:,i], (n_h, n_w))
        # get index of myocardium
        idx_myo = np.array(np.where(roi.segmentsResampled[i]['non-MI']>0)).T
        if len(idx_myo) > 0: # has myo segment
            if roni is not None: # remove the pixels in roni if roni is not None
                # Depend de la version de opencv : (4.5.1.48)==>tuple(endoCenters)
                #                                  (>= 4.5  )==>tuple(np.float32(idx_myo[j,:]))
                idx_myo = np.array([idx_myo[j] for j in range(idx_myo.shape[0]) \
                                    if cv.pointPolygonTest(np.float32(roni), 
                                                           tuple(np.float32(idx_myo[j,:])), 
                                                           False) < 0])
            ## radial
            contour_myo = roi.contoursResampled[i]['non-MI']
            coordinates[idx_myo[:,0],idx_myo[:,1],0,i], epi, endo = \
                compute_radials_inSlice(idx_myo, contour_myo, roi.endoCentersResampled[:,i])
                
            # ------ plot the auto detected epi & endo
            if flag_plot:
                if epi is not None and endo is not None:
                    fig_tmp, ax_tmp = plt.subplots(1,1, figsize=(4,4))
                    funPlot.scatter_contours_inSlice(ax_tmp, roi.contoursResampled[i], 
                                                     roi.labelColors, 
                                                     typeContShow='all')
                    ax_tmp.plot(endo[:,1], endo[:,0], 'r.', markersize=2)
                    ax_tmp.invert_yaxis()
                    ax_tmp.set(title='Detected endo plotted in red (slice {:d})'.format(i))
                    plt.pause(figurePause); plt.close()
            # ------ end plot ------------------------            
            
            ## angle cicumferential
            coordinates[idx_myo[:,0],idx_myo[:,1],1,i] = \
                compute_angles_inSlice(idx_myo, roi.originesResampled[:,i], 
                                        roi.endoCentersResampled[:,i])
            ## Z
            coordinates[idx_myo[:,0],idx_myo[:,1],2,i] = Z[i]
    return coordinates


def compute_radials_inSlice(idx_myo, contour_myo, endoCenter):
    """ compute radials of pixles in considered region given by idx_myo (within 1 slice)
        epi & endo are detected automatically if they are mixted in contour_myo
        Paramters: 
            idx_myo: ndarray [N,2] coords of pixels in considered region
            contour_myo: list of contours, instance of roi.contoursResampled[i]['non-MI']
            endoCenter: [2,]
        Returns:
            radial: [N,]
            epi: detected epi when mix with endo, [numPixelEpi,2] epicardium 
            endo: detected endo when mix with epi, [numPixelEndo,2] endocardium
    """
    if len(contour_myo) == 1:
        # Depend de la version de opencv : (4.5.1.48)==>tuple(endoCenters)
        #                                  (>= 4.5  )==>tuple(np.float32(funImg.find_endoCenter(contour_myo[0]))
        if cv.pointPolygonTest(np.float32(contour_myo[0]),  # if myo is in cresent shape
                               tuple(np.float32(funImg.find_endoCenter(contour_myo[0]))), 
                               False) < 0: 
            epi, endo = funImg.separate_epi_endo(contour_myo[0])
            dist_toEpi = funArray.apply_along_axis(funImg.shortestDist_point_to_contour, 
                                                   idx_myo, contour=epi)
            dist_toEndo = funArray.apply_along_axis(funImg.shortestDist_point_to_contour, 
                                                    idx_myo, contour=endo)
            radial = funArray.safe_divde(dist_toEndo, dist_toEndo + dist_toEpi)
        else:  # if epi exists but no endo
            epi = None; endo = None
            dist_toOrigines = np.linalg.norm(idx_myo - endoCenter, axis=1)
            dist_toEpi = funArray.apply_along_axis(funImg.shortestDist_point_to_contour, 
                                                   idx_myo, contour=contour_myo[0])
            radial = funArray.safe_divde(dist_toOrigines, dist_toOrigines + dist_toEpi)
    elif len(contour_myo) == 2: # both epi endo exist
        epi = None; endo = None
        dist_toEpi = funArray.apply_along_axis(funImg.shortestDist_point_to_contour, 
                                               idx_myo, contour=contour_myo[0])
        dist_toEndo = funArray.apply_along_axis(funImg.shortestDist_point_to_contour, 
                                                idx_myo, contour=contour_myo[1])
        radial = funArray.safe_divde(dist_toEndo, dist_toEndo + dist_toEpi)
    else:
        raise ValueError("sth wrong, roi.contoursResampled" + \
                          "has more than 2 contours or no contour")
    return radial, epi, endo

def compute_angles_inSlice(idx_myo, origine, endoCenter):
    """ compute angles of pixles in considered region given by idx_myo (within 1 slice)
        Paramters: 
            idx_myo: ndarray [N,2] coords of pixels in considered region
            origine: [2,] 
            endoCenter: [2,]
        Returns:
            angles: [N,] corresponding angles of idx_myo
    """
    vect_origine = origine - endoCenter
    ang_origine = np.arctan2(vect_origine[0], vect_origine[1]) 
    vect_myo = idx_myo - endoCenter
    angles = ang_origine - np.arctan2(vect_myo[:,0], vect_myo[:,1]) 
    angles[angles<0] += 2*np.pi # make sure all positive count from origine 
    return angles

def compute_z(USliceLocation, spacingZ, idx_apex, idx_valve):
    """ compute coordinate Z
        Paramters: 
            USliceLocations: ndarrray [numSlices,], location of slices 
            spacing: float, spacing between slices
            idx_apex: int/float
            idx_valve: int/float
        Returns:
            Z: ndarray [numSlices,], normalized z coordinate values
    """
    if np.any([np.isnan(idx_apex), np.isnan(idx_valve)]):
        raise ValueError('Attribute idxes of roi instance should be given before compute coordinate Z')
    if idx_apex < 0:
        z_apex = USliceLocation[0] + spacingZ * idx_apex
    else:
        z_apex = USliceLocation[np.floor(idx_apex).astype(int)] + spacingZ * (idx_apex-np.floor(idx_apex))
    z_range = (idx_valve - idx_apex) * spacingZ
    if spacingZ > 0:
        if z_range > 0:
            Z = (USliceLocation - z_apex)/z_range # Z can be <0 or >1
            Z[np.abs(Z-0) < np.abs(0.1/z_range)] = 0.0 # make sure the valve is 0 
            Z[np.abs(Z-1) < np.abs(0.1/z_range)] = 1.0 # make sure valve is 1 but not 0.9999999 which will effect interpolation
        elif z_range < 0:
            Z = np.nan; warnings.warn('valve is lower than apex.')
        else:
            Z = np.nan; warnings.warn('valve and apex lay in the same slice.')
    elif spacingZ < 0:
        if z_range < 0:
            Z = (USliceLocation - z_apex)/z_range # Z can be <0 or >1
            Z[np.abs(Z-0) < np.abs(0.1/z_range)] = 0.0 # make sure the valve is 0 
            Z[np.abs(Z-1) < np.abs(0.1/z_range)] = 1.0 # make sure valve is 1 but not 0.9999999 which will effect interpolation
        elif z_range > 0:
            Z = np.nan; warnings.warn('valve is lower than apex.')
        else:
            Z = np.nan; warnings.warn('valve and apex lay in the same slice.')    
    else:
        raise ValueError('Z spacing is 0')
    return Z


def ravel_coordinates(coordinates):
    """ ravel the coordinates 
        Parameters: 
            coordinates: [N_H, N_W, 3, numSlices]
        Returns: 
            coordinates_raveled: raveled coordinates [3, numCoordinates] ndarray
    """
    rad = coordinates[:,:,0,:].ravel()
    ang = coordinates[:,:,1,:].ravel()
    z   = coordinates[:,:,2,:].ravel()
    coordinates_raveled = np.array([rad, ang, z]).T
    return coordinates_raveled

def get_valid_coordinate_vals(coordinates):
    """ get valid coordinates values
        Parameters: 
            coordinates: [N_H, N_W, 3, numSlices]
        Return:
            radius_valid: [n_validRadVals,] ndarray, valid radius values
            angles_valid: [n_validAngVals,] valid angle values
            z_valid: [n_validZVals,] valid z values
    """
    coords_ravel = ravel_coordinates(coordinates)
    coords_ravel = coords_ravel[~(np.isnan(coords_ravel).any(axis=1))]
    radius_valid = np.unique(coords_ravel[0,:])
    angles_valid = np.unique(coords_ravel[1,:])
    z_valid      = np.unique(coords_ravel[2,:])
    return radius_valid, angles_valid, z_valid 

def get_data_with_coords(data, coordinates):
    """ Get data that with valide coordinate values, data keeps the same shape 
        while the values without coordinates will be set to nan
        Parameters: 
            coordinates: [N_H, N_W, 3, numSlices]
        Returns:
            data_withcoords: data which has it's values without valid coordinates set to nan.
    """
    if not (np.array(data.shape) == np.array(coordinates.shape)[[0,1,-1]]).all():
        raise ValueError('The shape of coordinates does not match the data')
    data_withcoords = np.full(np.shape(data), fill_value=np.nan) 
    idx_notnan = np.where(~(np.isnan(ravel_coordinates(coordinates)).any(axis=1)))[0]
    data_withcoords.ravel()[idx_notnan] = data.ravel()[idx_notnan]
    data_withcoords = data_withcoords.reshape(np.shape(data)) 
    return data_withcoords

def get_z_vals(coordinates):
    """ Get z values of each slices
        Parameters: 
            coordinates: [N_H, N_W, 3, numSlices]
        Returns:
            z_values: [numSlices,] ndarray of z value of each slice
    """
    z_values = [np.nan if np.all(np.isnan(coordinates[:,:,-1,i])) else \
              np.nanmax(coordinates[:,:,-1,i]) for i in range(coordinates.shape[-1])]
    return z_values

def get_origine_from_angle(contours, endoCenters, angle=np.pi*2/3):
    """ get origine on the epi contour
        Parameters:
            contours: roiObj.contours instance 
            endoCenter: [numSlices,2] ndarray, location of endoCenters
            angle: origine angle from est (counterclockwise)
    """
    origines = np.zeros((2,len(contours)))
    angle = angle-np.pi/2 # angle count from north
    for i in range(len(contours)):
        radial = np.linalg.norm(contours[i]['non-MI'][0][0] - endoCenters[:,i]) # radial of epi
        origines[:,i] = endoCenters[:,i] - radial*np.array([np.cos(angle), np.sin(angle)])
    return origines

def get_euclid_from_polar(angle, radial, center=np.array([0,0]), origineAngle=0, yx=False):
    """ get euclidian coordinates from polar coordinates
        Parameters:
            angle: float, the angle from origineAngle (counterclockwise)
            radial:float or [N,] ndarray of radials
            center: [2,] ndarray of the height and width of center
            origineAngle: the starting angle
            yx: if Ture return yx coordinates (origine at southwest) instead of hw (origine at northwest)
                x = w, y increases from bottom to top, h increases from top to bottom
        Return:
            hw: [N,2] ndarray of height and width or yx coordinates
    """
    if yx:
        theta = origineAngle + angle
        hw_gain = np.array([np.sin(theta), np.cos(theta)])
    else:
        theta = origineAngle + np.pi/2 + angle 
        hw_gain = np.array([np.cos(theta), np.sin(theta)]) 
    if isinstance(radial, float):
        hw = center + radial * hw_gain
    elif isinstance(radial, np.ndarray):
        hw = np.atleast_2d(center).T + np.atleast_2d(radial).T * hw_gain
    return hw

def get_segmentCenter(roi, typeSeg, samplingFactor):
    """ compute center of 3d segmentation (e.g. MI)
        Parameters:
            roi: instance of object funObj.RoiObj
            typeSeg: 'MI' or 'NR', type of segment need to find center
            samplingFactor: upsampling factor when got roi.segmentsResampled from roi.segments 
        return:
            miCenterPos: (,3) ndarray, postion of center 
            angCenter: float, angle of center in [0, 2pi]
    """
    miPos = np.vstack([np.insert(np.argwhere(roi.segments[i][typeSeg]>0),2,values=i,axis=1) \
                        for i in range(len(roi.segments))]) # segment positions
    miCenterPos = miPos[np.argmin((pairwise_distances(miPos)).mean(axis=0)),:] # center = the point insegment closest to all the other points
    miCenterPos[0:2] = (miCenterPos[0:2]-1)*samplingFactor + 1 # get the upsampled center postion at upsampled segments
    angCenter = roi.coordinates[miCenterPos[0],miCenterPos[1],1,miCenterPos[2]] # angle of the center
    if np.isnan(angCenter): # if upsampled center drops in nan region, then find the closest not nan postion instead
        posCoord = np.argwhere(~np.isnan(roi.coordinates[:,:,1,miCenterPos[2]]))
        miCenterPos[0:2] = posCoord[np.argmin(np.linalg.norm(posCoord - miCenterPos[0:2], axis=1))]
        angCenter = roi.coordinates[miCenterPos[0],miCenterPos[1],1,miCenterPos[2]]        
    return miCenterPos, angCenter

















