#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:52:14 2020

@author: zheng
"""

#import warnings
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../') 

import PolarLV.common.funIO as funIO
import PolarLV.common.funImg as funImg
import PolarLV.common.funObj as funObj
import PolarLV.common.funPlot as funPlot
import PolarLV.common.funAlign as funAlign 


class CorePolarLV():
    
    def __init__(self, ref=None, dcm=None, roi=None):
        """ Core processes to normalize foramted data and segmentation 
            Paramters:
                ref: reference, instance of polarLV.common.funObj.RefObj
                dcm: formated data, instance of polarLV.common.funObj.DicomObj
                roi: formated segmentations, instance of polarLV.common.funObj.RoiObj
        """
        self.ref = ref
        self.dcm = dcm
        self.roi = roi
        if ref is not None and not isinstance(ref, funObj.RefObj): 
            raise ValueError('ref should be instance of polarLV.common.funObj.RefObj') 
        if dcm is not None and not isinstance(dcm, funObj.DicomObj): 
            raise ValueError('ref should be instance of polarLV.common.funObj.DicomObj')   
        if roi is not None and not isinstance(roi, funObj.RoiObj): 
            raise ValueError('ref should be instance of polarLV.common.funObj.RoiObj') 
            
        self.numSlices = len(dcm.USliceLocation)
        if self.roi.segments is None: # add segmentations if missing
            self.roi.add_segments(dcm.data.shape[0:2])
        
    def figure_commun_setting(self, labelColors=None, typeSegShow=None, typeContShow=None):
        """ common setting for figures 
                (not necessry, the parameters can be specified when calling the plotting functions)
            paramters: 
                labelColors: dict, defines colors of different types of contours and segments
                typeSegShow: list, contains the types of segments to show
                typeContShow: list, contains the types of contours to show            
        """
        self.labelColors = labelColors
        self.typeSegShow = typeSegShow
        self.typeContShow = typeContShow
        return
     
    def user_locate_origine_myoOpening(self, labelColors = None, 
                                       typeSegShow = None, typeContShow = None):
        """ Guide user to locate origine and myocardium openings of MRI slice figures
                registrate them in self.roi 
            parameters: 
                labelColors: dict, defines colors of different types of contours and segments
                    if None, it will be assined CorePolarLV.labelColors 
                typeSegShow: list, contains the types of segments to show
                    if None, it will be assined CorePolarLV.typeSegShow
                typeContShow: list, contains the types of contours to show 
                    if None, it will be assined CorePolarLV.typeContShow 
        """
        origines    = np.full((2,self.numSlices), np.nan)  # apex to base
        myoOpenings = np.full((2,2,self.numSlices), np.nan)
        # show all slices and segments
        fig_org, ax_org = self.slices_image_2d(labelColors = labelColors, 
                                               typeSegShow = typeSegShow, 
                                               typeContShow = typeContShow, 
                                               titles = ['slice: {:d}, '.format(i) + \
                                                         'loc: {:.2f}'.format(self.dcm.USliceLocation[i]) \
                                                         for i in range(self.numSlices)])
        print('\n================================================== ',
              '\nRequire user inputs of LV-RV junction and myo openings for each slice.' +
              '\n - Please follow the instruction on the interactive slice figure,' +
              '\n - Press enter to move to the next slice if LV-RV junction does not present on the current slice, ' +
              '\n - if you have no idea where are junctions and openings, see ./samples/user_input_example.')
        for i in reversed(range(0, self.numSlices)):
            print(' \nslice {:d}'.format(i))
            ## show each slice for user to input origine and myo openings
            fig, ax = self.slices_image_2d(labelColors = labelColors, 
                                           typeSegShow = typeSegShow, 
                                           typeContShow = typeContShow, 
                                           idSlice = i,
                                           figsize = (5,4),
                                           suptitle= 'Slice {:d}'.format(i))
            # interact with figure to get origines & myoOpenings
            origines[:,i], myoOpenings[:,:,i] = funImg.locate_origine_myoOpening(fig)
            plt.close(fig)
        ## add to roi object the origined and myoOpenings 
        self.roi.add_origines_myoOpenings(origines, myoOpenings, adjust=0) 
        return
    
    def user_locate_apex_base(self, labelColors = None, 
                              typeSegShow = None, typeContShow = None,
                              ):
        """ Guide user to locate index of apex and base slices
                to help easily estimating them, a long-axis chamber view is estimated
                during the process
            parameters: 
                labelColors: dict, defines colors of different types of contours and segments
                    if None, it will be assined CorePolarLV.labelColors 
                typeSegShow: list, contains the types of segments to show
                    if None, it will be assined CorePolarLV.typeSegShow
                typeContShow: list, contains the types of contours to show 
                    if None, it will be assined CorePolarLV.typeContShow 
        """
        print('\n==================================================',
              '\nRequire user to input the slice indexes of apex and base.' +
              '\n - Please follow the instruction printed in the console.' +
              '\n - if you have no idea where are apex and base, see ./samples/user_input_example.')
        chamberView, fig_chamb, ax_chamb = self._chamber_view(labelColors, typeSegShow , typeContShow)
        ## add lines to show slices with origine records (in yellow line)
        idx_hasOrigine = np.where(~np.isnan(self.roi.origines[0,:]))[0]
        ax_chamb.hlines(self.numSlices-1-idx_hasOrigine[0], 0, chamberView.shape[1], 
                        colors='y', linewidth=2)
        ax_chamb.hlines(self.numSlices-1-idx_hasOrigine[-1], 0, chamberView.shape[1], 
                        colors='y', linewidth=2)
        funPlot.tellme('The interval between yellow lines is with origines inputs') 
        plt.pause(0.1)
        ## add to roi objects, the user input estimation of apex and base
        funPlot.tellme('\nNow input the estimation of the slice numbers of apex and valve\n' +
                       ' you can refer to the chamber view and 2d slices plot.') 
        idx_apex, idx_valve = funIO.get_numSlice_apex_valve()
        self.roi.add_index_apex_valve(idx_apex, idx_valve)
        
        ## add lines to show selected apex and valve (in red dotted line)
        ax_chamb.hlines(self.numSlices-1-self.roi.idxes['apex'], 0, chamberView.shape[1], 
                        colors='r', linewidth=2, linestyles='dashed')
        ax_chamb.hlines(self.numSlices-1-self.roi.idxes['valve'], 0, chamberView.shape[1], 
                        colors='r', linewidth=2, linestyles='dashed')    
        funPlot.tellme('apex and valve slices in red dotted line') 
        plt.pause(0.1)
        # adjust the origines and myoOpenings, above base and below apex are abandoned
        self.roi.add_origines_myoOpenings(self.roi.origines, self.roi.myoOpenings, adjust=1) 
        # fig_chamb.savefig('chamber_view.png') 
        return
    
    def add_endoCenter(self, mtdEndoCenter_mixEndoEpi = 'copy'):
        """ Estimate endocardium center 
            parameters:
                mtdEndoCenter_mixEndoEpi: str, the method to deal with the case where endo, 
                    epi contours are not separated, wich often forms a crescent myo shape.
                        'copy', will copy the endoCenter of the nearest slice 
                        'convexHull', will take the centroid of the convexHull of the crescent
                        as the endoCenter
        """
        self.roi.add_endoCenters(mtdEndoCenter_mixEndoEpi=mtdEndoCenter_mixEndoEpi) 
        print('\n Endo centers added')
        return
    
    def up_sampling(self, samplingFactor = 4,
                    mtdUpsampling = 'linear',
                    mtdInterpContour = None):
        """ Upsampling to increase the resolution of dicom data and segmentations
            parameters:
                sampling Factor: int, for a dimention, the upsampled resolution will be 
                    (original resolution - samplingFactor) * 4 + 1
                mtdUpsampling: string, iterpolation method for upsampling dicom image, 
                    options are the same as "method" of scipy.interpolate.griddata
                mtdInterpContour: interpolation method for upsampling contours
                    None = linearly interpolated
        """
        print('\nUp sampling (factor: {:d}) ... '.format(samplingFactor))
        # up sampling data, and update spacing
        self.dcm.resample(samplingFactor, mtdInterp = mtdUpsampling) 
        # up sampling contours, segments, origines, myoOpenings, endoCenters if exist
        self.roi.resample(samplingFactor, mtdInterp = mtdInterpContour) 
        return
    
    def compute_coordinates(self, debugPlot=0, figurePause=4):
        """ Compute polar coordinates of each pixel in region of interest of MRI slices
            paramters:
                debugPlot: boolean, if 1 print some figures to add debug
                figurePause: the time to plt.pause() for each debug figure
        """
        print('\nCompute polar coordinates ... ')
        self.roi.add_coordinates(self.dcm, flag_debugPlot=debugPlot, figurePause=figurePause) # compute polar coordinates
        return
    
    def alignment(self, mtd = 'linear', idx_slice_ref = None, fill_value = 0, removeMissingAngles = 1):
        """ Aligne MRI data and segmentations to reference
            parameters:
                mtd: string, 'linear' (faster), 'cubic'
                fill_value: filled value for extrapolation
                removeMissingAngles: boolean, if = 1, the open angle (caused by myo opening) 
                    is removed from normalized data
            return:
                data_normalized: ndarray, [Height, Width, slice number] normalized MRI data
                segments_normalized: list of dict, normalized segmentations
        """
        print('\nAlignment (may take several minutes) ... ')
        plt.close('all')

        data_normalized, segments_normalized, _ = funAlign.alignment(self.dcm, self.roi, 
                                                                self.ref.roi, 
                                                                idx_slice_ref = idx_slice_ref,
                                                                mtd = mtd, 
                                                                fill_value = fill_value, 
                                                                removeMissingAngles = removeMissingAngles)

        return data_normalized, segments_normalized
        
    def slice_image_2d_normalized(self, data_normalized, segments_normalized, 
                                  mode17seg = 1, labelColors = None, figsize = (10,7),
                                  typeSegShow = None, typeContShow = None,
                                  cmapData = 'gray', vmin = None, vmax = None):
        """ Show normalized 2d slices images and segmentations
            parameters:
                data_normalized: ndarray, normalized MRI data, output of CorePolarLV.alignment()
                segments_normalized: list of dict, normalized segmentations, output of CorePolarLV.alignment()
                mode17seg: boolean, if = 1, will show the data in 17 segments heart mode
                labelColors: dict, defines colors of different types of contours and segments
                    if None, it will be assined CorePolarLV.labelColors 
                typeSegShow: list, contains the types of segments to show
                    if None, it will be assined CorePolarLV.typeSegShow
                typeContShow: list, contains the types of contours to show 
                    if None, it will be assined CorePolarLV.typeContShow 
                cmap: color map of normalized MRI data
            return:
                fig, ax
        """
        labelColors, typeSegShow , typeContShow = \
            self._get_figure_Setting(labelColors, typeSegShow , typeContShow)
        if mode17seg:
            model_17seg = funObj.Model17SegObj(self.ref)
            data = model_17seg.fit(data=data_normalized)
            segments = model_17seg.fit(segments=segments_normalized)
            origines = model_17seg.origines
        else:
            data = data_normalized
            segments = segments_normalized
            origines = self.ref.roi.originesResampled
        fig, ax = funPlot.plot_2d_data_contours_segments(data = data,
                                                         labelColors = labelColors,
                                                         contours = self.ref.roi.contoursResampled,
                                                         segments = segments,
                                                         typeContShow = typeContShow,
                                                         typeSegShow = typeSegShow,
                                                         cmap=cmapData, 
                                                         figsize = figsize,
                                                         vmin = vmin, vmax = vmax,
                                                         suptitle = 'Normalized data and segments',
                                                         titles = ['z: {:.3f}, '.format(i) \
                                                                  for i in self.ref.roi.get_z_vals()]
                                                         )
        funPlot.show_origine(ax, origines)
        if mode17seg:
            for i in range(self.ref.numSlices):
                funPlot.show_17seg_lineSeparations_inSlice(ax[np.unravel_index(i, ax.shape)], 
                                                           model_17seg.get_separations_inSlice(i)[1])
        return fig, ax
    
    def show_coordinates(self, savefig = 0, folderFigure=None):
        """ plot coordinates got by CorePolarLV.compute_coordinates()
                of each slices and save them to file
            parameters: 
                savefig: boolean, if 1, save the coordinates figures
                folderFigure: the folder in which a subfolder called "coordinates"
                    will create to save plotted coordinates
        """
        plt.ioff()
        for i in range(self.numSlices): # plot coords
            if not np.all(np.isnan(self.roi.coordinates[:,:,:,i])): # has coordiantes value in slice
                fig_coord, ax_coord = \
                    funPlot.plot_coordinates_inSlice(self.roi.coordinates[:,:,:,i], 
                                                     suptitle='coordinates (slice: {:d})'.format(i),
                                                     origines=self.roi.originesResampled[:,i], 
                                                     myoOpenings=self.roi.myoOpeningsResampled[:,:,i], 
                                                     endoCenters=self.roi.endoCentersResampled[:,i])
                if savefig:
                    if folderFigure is not None:
                        folderFigureCoords = os.path.join(folderFigure, 'coordinates') # sub folder to save coords image
                        if not os.path.exists(folderFigureCoords):
                            os.makedirs(folderFigureCoords)
                        fig_coord.savefig(os.path.join(folderFigureCoords, \
                                                       'coordinates_slice_{:d}'.format(i) + '.png'))
                    else:
                        raise ValueError('Can not save figs, savefig = 1 while folderFigure = None')
                # plt.pause(1); plt.close()
        plt.ion()
        return
    
    def slices_image_2d(self, labelColors = None, typeSegShow = None, typeContShow = None, 
                        idSlice = None, resampled = 0, 
                        showOrigine = 0, showMyoOpening = 0, showEndoCenter = 0,
                        figsize = (10,7), 
                        suptitle = '2D view contours in slices (apex to base)',
                        titles = None,
                        cmap = 'gray', vmin = None, vmax = None):
        """ Show 2d slices images and segmentations (if provided)
            parameters:
                 labelColors: dict, defines colors of different types of contours and segments
                     if None, it will be assined CorePolarLV.labelColors 
                typeSegShow: list, contains the types of segments to show
                    if None, it will be assined CorePolarLV.typeSegShow
                typeContShow: list, contains the types of contours to show 
                    if None, it will be assined CorePolarLV.typeContShow 
                idSlice: None or int, if None, all slices will be plotted, 
                    if int, only the slice indexed will be plotted
                resampled: boolean, if = 1, resampled data and segmentations are shown 
                    instead of original ones
                showOrigine: boolean, if = 1, plot user input origines 
                showMyoOpening: boolean, if = 1, plot user input myocardium openings
                showEndoCenter: boolean, if = 1, plot estimated endocardium centers 
                cmap: color map of MRI data
            return:
                fig, ax
        """
        labelColors, typeSegShow , typeContShow = \
            self._get_figure_Setting(labelColors, typeSegShow , typeContShow)
        if resampled:
            dcmData  = self.dcm.dataResampled
            contours = self.roi.contoursResampled
            segments = self.roi.segmentsResampled
            origines = self.roi.originesResampled
            myoOpenings = self.roi.myoOpeningsResampled
            endoCenters = self.roi.endoCentersResampled
        else:
            dcmData  = self.dcm.data
            contours = self.roi.contours
            segments = self.roi.segments
            origines = self.roi.origines
            myoOpenings = self.roi.myoOpenings
            endoCenters = self.roi.endoCenters
        if idSlice is not None:
            dcmData  = dcmData[:,:,idSlice]
            contours = contours[idSlice]
            segments = segments[idSlice]
        fig, ax = funPlot.plot_2d_data_contours_segments(dcmData, 
                                                         labelColors,
                                                         contours, 
                                                         segments,
                                                         typeContShow = typeContShow, 
                                                         typeSegShow = typeSegShow,
                                                         cmap = cmap,
                                                         figsize = figsize,
                                                         vmin = vmin, vmax = vmax,
                                                         suptitle = suptitle,
                                                         titles = titles
                                                         )
        if showOrigine:
            funPlot.show_origine(ax, origines[:,idSlice] if idSlice is not None else origines)  
        if showMyoOpening:
            funPlot.show_myoOpening(ax, myoOpenings[:,:,idSlice] if idSlice is not None else myoOpenings)  
        if showEndoCenter:
            funPlot.show_endoCenter(ax, endoCenters[:,idSlice] if idSlice is not None else endoCenters) 
        # plt.pause(0.1)
        return fig, ax
    
    def contours_3d(self, typeContShow = 'all', labelColors = None, 
                    title ='3D view contours in slices'):
        """ plot 3d contours of segmentations
            paramters:
                typeContShow: 'all' or list of types, if = 'all', all type of contours will be plotted
                labelColors: dict, defines colors of different types of contours and segments
                     if None, it will be assined CorePolarLV.labelColors 
        """
        labelColors, _, _ = self._get_figure_Setting(labelColors = labelColors)
        if labelColors is None:
            raise ValueError('Parameter labelColors should be given')
        fig, ax = funPlot.plot_3d_contours(self.roi.contours, 
                                           self.dcm.USliceLocation, 
                                           labelColors,
                                           typeContShow,
                                           suptitle=title)
        # plt.pause(0.1)
        return fig, ax
    
    def _chamber_view(self, labelColors, typeSegShow , typeContShow):
        """ estimate long-axis chamber view
            return:
                chamberView: ndarray of estimated chamber view
                fig_chamb, ax_chamb
        """
        ## show the mid-slice of which have origine record
        idx_midSlice = np.median(np.where(~np.isnan(self.roi.origines[0,:]))[0]).astype(int)
        fig_mid, ax_mid = self.slices_image_2d(labelColors, typeSegShow, typeContShow,
                                               idSlice = idx_midSlice,
                                               figsize = (8,6.5),
                                               suptitle= 'Mid Slice {:d}'.format(idx_midSlice)) 
        ## get a line across RV-LV at mid-slice
        pts_line = funImg.interact_draw_line(fig_mid, ax_mid)
        ## extend the line to the edge of figure
        pts_line = funImg.extend_line_to_edge(pts_line, self.dcm.data[:,:,idx_midSlice].T.shape, ax_mid) 
        mask_line = funImg.fill_line(pts_line, self.dcm.data[:,:,idx_midSlice].shape)
        
        ## show the long axis chamber view (by hstacking the cross line of each slice)
        chamberView = np.zeros((self.numSlices, mask_line.sum()))
        for i in range(0, self.numSlices):
            chamberView[i, :] = self.dcm.data[:,:,i].ravel('F')[mask_line.ravel('F')>0]
        fig_chamb, ax_chamb = funPlot.plot_chamber(np.flipud(chamberView))
        plt.close(fig_mid) # stop showing the middle slice
        return chamberView, fig_chamb, ax_chamb
    
    def _get_figure_Setting(self, labelColors = None, 
                            typeSegShow = None, typeContShow = None):
        if labelColors is None and hasattr(self,'labelColors'):
            labelColors = self.labelColors
        if typeSegShow is None and hasattr(self,  'typeSegShow'):
            typeSegShow = self.typeSegShow
        if typeContShow is None and hasattr(self, 'typeContShow'):
            typeContShow = self.typeContShow
        return labelColors, typeSegShow, typeContShow
    
        
