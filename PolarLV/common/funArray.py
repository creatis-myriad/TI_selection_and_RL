#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:04:07 2020

@author: zheng
"""
import sys
sys.path.append('../../')

import numpy as np
import PolarLV.common.funDict as funDict


def isPolygon(array):
    """ Check if array is polygon
        Paramters:
            array: numpy array [N,2]
    """
    if array.shape[1] != 2:
        raise TypeError('Input should be numpy array with shape [N,2].')
    flag_polygon = False
    if array.shape[0] > 2:    # circle should have more than 3 points
        diff_array = np.diff(array, axis=0).astype(float)
        diff_array[diff_array[:,0] == 0, 0] = 10**(-10)
        if np.any(diff_array[:,0] == 0):
            print(diff_array[:,0])
        gradient = diff_array[:,1]/diff_array[:,0]
        if np.any(gradient != gradient[0]):  # circle shold not be a line
            flag_polygon = True
    return flag_polygon

def mask_to_index(array, maskVal = None):
    """ return the indexes of locations in a mask array
        Paramters: 
            array: numpy array
            maskVal: the mask value, if exist take the idex of the maskVal
                else take index > 0
        Returns:
            idx: np array with locations
        e.g. array =[[0,1,1],   idx = [[0,1],
                     [1,8,0]]          [0,2],
                                       [1,0]]
    """
    if maskVal is not None:
        idx = np.vstack(np.where(array == maskVal)).transpose()
    else:
        idx = np.vstack(np.where(array > 0)).transpose()
    return idx

def apply_along_axis(fun, dataMatrix, axis=1, numFunOut=1, **kwargs):
    """ apply function along axis of dataMatrix
            e.g. fun=np.mean, dataMatrix=np.array([[1,2,3]]), axis=1
            return 2
        Paramters: 
            fun: function (can have several outputs)
            dataMatrix: data matrix, if size <=0, return empty array
            axis: the axis that function will apply on
            numFunOut: function outputs number
        Returns:
            out: ndarray of result
    """
    if type(dataMatrix) is not np.ndarray:
        raise TypeError('dataMatrix should be numpy array')
    if dataMatrix.size > 0:
        funOut = np.apply_along_axis(fun, axis=axis, arr=dataMatrix, **kwargs)
        if numFunOut > 1:
            if funOut.shape[1] != numFunOut:
                raise ValueError('number of function outputs does not equal to numFunOut')  
            else:
                # print(funOut)
                funOut = tuple(funOut.T)
    else:
        if numFunOut > 1:
            funOut = (np.array([]),)*numFunOut
        else:
            funOut = np.array([])
    return funOut

def closest_pointInContour_to_point(contour, point):
    """ find closest point in the contour to a given point 
        Paramters: 
            contour: ndarray [N,2]
            point: ndarray [2,]
        Returns:
            closest_pointInContour: the point on the contour which is the closest to the point
    """
    dists = np.linalg.norm(contour - point, axis=1)
    idx = np.where(dists == dists.min())[0][0]
    closest_pointInContour = contour[idx, :]
    return closest_pointInContour, idx

def safe_divde(numerator, denominator, addedDiviate=10**(-10)):
    """ make sure division can be safely processed
        Paramters: 
            numerator: [N,] ndarray
            denominator: [N,] ndarray
            addedDiviate: added value to denominator if 0 encountered
        Returns:
            division result
    """
    denominator[denominator==0] += addedDiviate
    return numerator/denominator

def removeKey_from_ListOfDict(array, keys):
    """ remove a keys and values from each elements of a dict array
    """
    if not isinstance(array, list):
        raise TypeError('Input must be a list')
    if not isinstance(keys, list):
        raise TypeError('Inpurt keys must be a list of strings')
    for i in range(len(array)):
        for key in keys:
            array[i] = funDict.remove_key(array[i], key)
    return array

def idxElements_of_sublistInList(L, subL):
    idx = []; commonElem = []
    for elem in subL:
        if elem in L:
            commonElem.append(elem)
            idx.append(L.index(elem))
    return idx, commonElem

def scale(array, scale=(0,1), originScale=None):
    """ scale data to [scaleMin, scaleMax]
        parameters: 
            array: ndarray
            scaleMin, scaleMax: interval borders
            originScale: [scaleMin, scaleMax], the scale of original array considered 
                 if None, use the scale of the input array 
    """
    array = np.array(array)
    if originScale is not None:
        array = (array-min(originScale))/(max(originScale)-min(originScale))
    else:    
        if np.nanmax(array) - np.nanmin(array) > 0:
            array = (array-np.nanmin(array))/(np.nanmax(array)-np.nanmin(array))
        else:
            raise ValueError('The original scale of input array is 0.')
    return array

