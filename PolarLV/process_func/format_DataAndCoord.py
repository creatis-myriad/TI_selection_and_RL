#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:51:31 2021

example of preprocess which formats MRI (Dicom) data and corresponding segmentations

@author: zheng
"""

import sys, os
sys.path.append('../')

import pickle
from PolarLV.core.core_preprocess import CorePreprocess
from PolarLV.core.core_polarlv import CorePolarLV



def process(dcmFolder, 
            roiFile, 
            labelColors = {'non-MI':[[0,1,0],[1,0,0]], # dict of colors representing different segmentations
                           'MI': [0,0,1],              # 'non_MI': myocardium color, epi:[0], endo:[1]
                           'NR': [1,0,1]},             # 'MI': infarct, 'NR': No Reflow (MVO),     
            folderFormatedData = None,
            folderFigure = None,
            inverseDcm = 0,
            checkSpacing = 1,
            display = True,
            format_nifti = False,
            ):

    if format_nifti:
        roi, slices_to_remove = CorePreprocess.format_roi_nifti(roiFile = roiFile, 
        ) 
        dcm = CorePreprocess.format_dcm_nifti(dcmFile = dcmFolder,
                                        inverseDcm = inverseDcm, 
                                        checkSpacing = checkSpacing,
                                        slices_to_remove = slices_to_remove
        )
    else :
        # dcm (dicom) object collects info provided by dicom files
        dcm = CorePreprocess.format_dcm(dcmFolder = dcmFolder,
                                        inverseDcm = inverseDcm, 
                                        checkSpacing = checkSpacing)
        
        # roi object collects info of segmentations and post processing results 
        #  (e.g. locations of origines, myoOpenings, apex slice idx etc)
        #  depending on the segmentation file, this function may need to be implemented by user ...
        roi = CorePreprocess.format_roi(roiFile = roiFile, 
                                        shapeDcm = dcm.data[:,:,0].shape, # shape of each slice
                                        numSlices = dcm.data.shape[-1])   # total slices number
                                        
    # crop the data to focus aroud roi
    dcm, roi = CorePreprocess.crop_dcm_roi(dcm, roi, cropRoiEdge = 20)


    # save the formated data 
    if folderFormatedData is not None:
        CorePreprocess.pickle_formated_data(folderFormatedData, 'dcm.pkl', dcm)
        CorePreprocess.pickle_formated_data(folderFormatedData, 'roi.pkl', roi)
        print('Formated data has been saved to {:s}\n'.format(folderFormatedData))
             
    ## plots
    plv = CorePolarLV(dcm=dcm, roi=roi)
    # 2d image & contours
    if display :
        fig, ax = plv.slices_image_2d(labelColors = labelColors, 
                                    typeContShow = ['non-MI','MI','NR'],
                                    titles = ['slice: {:d}, loc: {:.2f}'.format(i, dcm.USliceLocation[i]) \
                                    for i in range(dcm.data.shape[-1])])
        # 3d contours
        fig_3d, ax_3d = plv.contours_3d(typeContShow = 'all', 
                                        labelColors=labelColors)
    
    if folderFigure is not None:   
        if not os.path.exists(folderFigure):
            os.makedirs(folderFigure)              
        fig.savefig(os.path.join(folderFigure, 'Data_Contour_2d.png'))
        fig_3d.savefig(os.path.join(folderFigure, 'Contour_3d.png'))
        print('Figures have been saved to {:s}\n'.format(folderFigure))
    return dcm, roi

