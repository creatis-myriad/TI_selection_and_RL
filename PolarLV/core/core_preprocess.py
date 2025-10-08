#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:13:41 2020

@author: zheng
"""

import sys, os
import numpy as np
import pandas as pd
import cv2 as cv
import pickle
import nibabel as nib
from pydicom import dcmread
sys.path.append('../../') 

import PolarLV.common.funObj as funObj
import PolarLV.common.funImg as funImg

class CorePreprocess():
    """ data preprocessing to format dicom files of short axis slices and segmentations 
    """
    
    def __init__(self):
        pass
        
    @staticmethod    
    def format_dcm(dcmFolder, inverseDcm=0, checkSpacing=1):
        """ Format dicom files 
            Paramters:
                dcmFolder: string, folder contains dicom files of heart slices
                inverseDcm: boolean, 1 = inverse the order of slices
                checkSpacing: boolean, 1 = check and adjust long axis spacing value  
            Return:
                dcm: object of funObj.DicomObj, formated dicom data
        """
        print('Dicom folder: {:s}'.format(dcmFolder))
        sliceList = sorted(os.listdir(dcmFolder))
        numSlices = len(sliceList)
        print('slices num: {:d}'.format(numSlices))
        ## create Dicom object 
        dcm = funObj.DicomObj(folder = dcmFolder,
                              UID = np.empty(numSlices, dtype=object),
                              USliceLocation = np.full(numSlices,fill_value=np.nan))
        flag_1stSlice = 1
        for i in range(0, numSlices):
            sliceFile = os.path.join(dcmFolder, sliceList[i])
            ds = dcmread(sliceFile)
            if flag_1stSlice == 1:
                dcm.data = np.zeros([ds.Rows, ds.Columns, numSlices])
                dcm.ImagePositionPatient = ds.ImagePositionPatient
                dcm.offset  = [0, 0, float(ds.SliceLocation)]
                dcm.spacing = list(np.float_(ds.PixelSpacing[:] + [ds.SliceThickness]))
            dcm.UID[i] = ds.file_meta.MediaStorageSOPClassUID # the same for all slices??
            dcm.USliceLocation[i] = float(ds.SliceLocation)
            if ds.PixelData is None:
                print('slice file {:s} has no pixel data!'.format(sliceList[i]))
                dcm.data[:,:,i] = 0
            else:
                dcm.data[:,:,i] = ds.pixel_array              
            flag_1stSlice = 0
        # check properties of dcm data & contours
        if inverseDcm:
            dcm.inverse_slices() # inverse slices
        if checkSpacing:
            dcm.check_zSpacing() # deal with wrong spacing registration 
        print('DICOM files have been formated\n')
        return dcm
    
    @staticmethod
    def format_roi(roiFile, shapeDcm, numSlices=None, labelColors=None):
        """ Format segmentation (roi) file 
                This function may not suit for your segmentation file. In this case, 
                write your own function to convert the segmentation to funObj.RoiObj.
            Paramters:
                roiFolder: string, segmentation file (.csv)
                shapeDcm: tuple, shape of each dicom slice
                numSlices: int, total slice number = dicom files number
                labelColor: dict, contains color of contours of segments, 
                    if given, it will be assigned to funObj.RoiObj
            Return:
                dcm: object of funObj.RoiObj, formated segmentations 
        """
        print('Roi folder: {:s}'.format(roiFile))
        labels = pd.read_csv(roiFile, skiprows=2,        # read labels
                             nrows=3, header=None).to_numpy()
        labels = dict((labels[i,1], labels[i,0]) \
                      for i in range(0,np.shape(labels)[0]))  # convert labels to dict
        deno = pd.read_csv(roiFile, skiprows=6, header=None,
                           names=['loc1','loc2','loc3','label','d1','d2','slice']) # read in denotations 
        ## create roi object 
        if numSlices is None:
            numSlices = max(deno['slice']) # in this case, numSlice <= corresponding dcm slices number
        roi = funObj.RoiObj(folder = roiFile,
                            points = len(deno),
                            labelColors = labelColors,
                            contours = [{key: [] for key in labels} for i in range(numSlices)]) 
        for i in range(0, numSlices):
            ## deal with contours
            slDeno = deno.loc[deno['slice'] == i]        
            for key, val in labels.items():
                zoneIdx = slDeno.loc[slDeno['label']>=val]
                if len(zoneIdx) > 0: # zone exists
                    canvas = np.zeros(shapeDcm)
                    canvas[tuple(zoneIdx['d2']), tuple(zoneIdx['d1'])] = 1
                    contours, hierarchy = cv.findContours(canvas.astype(np.uint8), 
                                                  cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                    contours, hierarchy = funImg.purify_contours(contours, hierarchy) # remove not circle contours
                    # !! The contours found by cv is based on X,Y coordinates
                    # imag [H, W, Z] but contour [Y, X, Z], we exchange the 1st 2 dims in contours
                    if len(contours) > 0: # contour found
                        contours = [np.fliplr(np.atleast_2d(cont.squeeze())) \
                                    for cont in contours] # do exchange of axis make contours [H, W]
                        if key == 'non-MI': # myocardium
                            if hierarchy[0].shape[0]>2:
                                raise ValueError('More than 2 contours found for "non-MI" ?')
                            elif hierarchy[0].shape[0]<2:
                                roi.contours[i][key] = contours
                            else:
                                if hierarchy[0][0,2]!=1: # contours[0] is endo
                                    contours.reverse()
                                roi.contours[i][key] = contours
                        else: # key in ['MI', 'NR']
                            roi.contours[i][key] = contours
        # add segments according to cotours
        roi.add_segments(canvasShape = shapeDcm)
        print('ROI file has been formated\n')
        return roi
    
    @staticmethod    
    def format_dcm_nifti(dcmFile, inverseDcm=0, checkSpacing=1, info_interp_slice=None, slices_to_remove=[]):
        """ Format dicom files 
            Paramters:
                dcmFile: string, file contains heart slices
                inverseDcm: boolean, 1 = inverse the order of slices
                checkSpacing: boolean, 1 = check and adjust long axis spacing value  
            Return:
                dcm: object of funObj.DicomObj, formated dicom data
        """
        print('Dicom folder: {:s}'.format(dcmFile))
        img = nib.load(dcmFile)
        header = img.header
        img_data = img.get_fdata()

        img_data = np.flip(img_data, axis=1)
        img_data = np.rot90(img_data, k=1, axes=(0, 1))

        if len(slices_to_remove) > 0:
            img_data = np.delete(img_data, slices_to_remove, axis=2)

        numSlices = img_data.shape[-1]
        print('slices num: {:d}'.format(numSlices))
        ## create Dicom object 
        dcm = funObj.DicomObj(
            folder = dcmFile,
            UID = np.empty(numSlices, dtype=object),
            USliceLocation = np.full(numSlices,fill_value=np.nan)
        )

        dcm.data = img_data
        dcm.spacing = header["pixdim"][1:4]
        dcm.offset  = [None, None, None]
        dcm.USliceLocation = np.arange(0, numSlices)*dcm.spacing[-1]

        if inverseDcm:
            dcm.inverse_slices() # inverse slices
        if checkSpacing:
            dcm.check_zSpacing() # deal with wrong spacing registration 
        print('DICOM files have been formated\n')
        return dcm
    
    @staticmethod
    def format_roi_nifti(roiFile, labelColors=None, info_interp_slice=None):
        """ Format segmentation (roi) file 
                This function may not suit for your segmentation file. In this case, 
                write your own function to convert the segmentation to funObj.RoiObj.
            Paramters:
                roiFolder: string, segmentation file (.csv)
                numSlices: int, total slice number = dicom files number
                labelColor: dict, contains color of contours of segments, 
                    if given, it will be assigned to funObj.RoiObj
            Return:
                dcm: object of funObj.RoiObj, formated segmentations 
        """
        print('Roi folder: {:s}'.format(roiFile))

        seg = nib.load(roiFile)
        header = seg.header
        seg_data = seg.get_fdata()
        points = len(np.where(seg_data > 0)[0])

        seg_data = np.flip(seg_data, axis=1)
        seg_data = np.rot90(seg_data, k=1, axes=(0, 1))
        numSlices = seg_data.shape[-1]

        # condition to remove slices with MVO
        if len(np.unique(seg_data)) == 5:
            slices_to_remove = []
            for i in range(numSlices):
                unique_labels_in_slice = np.unique(seg_data[:, :, i])
                if len(unique_labels_in_slice) == 5:
                    slices_to_remove.append(i)
            seg_data = np.delete(seg_data, slices_to_remove, axis=2)
            numSlices = seg_data.shape[-1]
        else: slices_to_remove = []

        seg_MI = np.copy(seg_data)
        seg_nonMI = np.copy(seg_data)

        seg_MI[np.where(seg_MI != 3)] = 0 # MI
        seg_nonMI[np.where(seg_MI == 3)] = 2
        seg_nonMI[np.where(seg_nonMI != 2)] = 0 # non-MI
        seg_NR = np.zeros(seg_nonMI.shape)

        D_seg = {
            "non-MI": seg_nonMI,
            "MI": seg_MI,
            "NR": seg_NR,
        }

        ## create roi object 
        if numSlices is None: numSlices = seg_nonMI.shape[-1]
        labels = ["non-MI", "MI", "NR"]
        roi = funObj.RoiObj(
            folder = roiFile,
            points = points,
            labelColors = labelColors,
            contours = [{key: [] for key in labels} for i in range(numSlices)]
        )
        
        for i in range(0, numSlices):       
            for key, val in D_seg.items():
                canvas = val[:,:,i]
                if np.any(canvas > 0) :
                    contours, hierarchy = cv.findContours(canvas.astype(np.uint8), 
                                                  cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                    contours, hierarchy = funImg.purify_contours(contours, hierarchy) # remove not circle contours
                    # !! The contours found by cv is based on X,Y coordinates
                    # imag [H, W, Z] but contour [Y, X, Z], we exchange the 1st 2 dims in contours
                    if len(contours) > 0: # contour found
                        contours = [np.fliplr(np.atleast_2d(cont.squeeze())) \
                                    for cont in contours] # do exchange of axis make contours [H, W]
                        if key == 'non-MI': # myocardium
                            if hierarchy[0].shape[0]>2:
                                raise ValueError('More than 2 contours found for "non-MI" ?')
                            elif hierarchy[0].shape[0]<2:
                                roi.contours[i][key] = contours
                            else:
                                if hierarchy[0][0,2]!=1: # contours[0] is endo
                                    contours.reverse()
                                roi.contours[i][key] = contours
                        else: # key in ['MI', 'NR']
                            roi.contours[i][key] = contours
        # add segments according to cotours
        roi.add_segments(canvasShape = seg_nonMI.shape[:2])
        print('ROI file has been formated\n')
        return roi, slices_to_remove
    
    @staticmethod
    def crop_dcm_roi(dcm, roi, cropRoiEdge = 20):
        """ Crop the formated dicom and roi object
            Parameters:
                dcm: instance of funObj.DicomObj
                roi: instance of funObj.RoiObj
                cropRoiEdge: float, [mm], the edge around the contour which will be cropped out
            Returns: 
                dcm: dcm_cropped
                dcm: roi_cropped
        """
        dcm_cropped, roi_cropped = funImg.crop_dcmRoi(dcm, roi, edge = cropRoiEdge)
        return dcm_cropped, roi_cropped
    
    @staticmethod
    def pickle_formated_data(folderFormatedData, pickleFileName, obj):
        """ Save formated data to pickle file
            Parameters:
                folderFormatedData: string, folder that the data will be pickled to 
                pickleFileName: string, file that the data will be pickled in 
                obj: the object to save
        """
        if not os.path.exists(folderFormatedData):
            os.makedirs(folderFormatedData)
        pickle.dump(obj, open(os.path.join(folderFormatedData, pickleFileName), 'wb')) 
        return
        
        