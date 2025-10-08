#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:19:51 2020

@author: zheng
"""

import sys
sys.path.append('../../')

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# import matplotlib.colors as clrs
import numpy as np
import PolarLV.common.funImg as funImg         
import PolarLV.common.funArray as funArray 

def plot_coordinates_inSlice(coordinates, 
                             figsize=(15,10), cmap='jet', suptitle='coordinates',
                             valueRanges = [[None,None], [None,None], [0,1]],
                             origines=None, myoOpenings=None, endoCenters=None):
    """ plot polar coordinates in 1 slice
        Paramters:
            valueRanges: [vmin,vmax] of each coordinate for imshow
    """
    plotSubTitles = ['radial', 'angle', 'z']
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for j in range(len(ax)):   
        ax[j].imshow(coordinates[:,:,j], cmap=cmap, 
                     vmin=valueRanges[j][0], vmax=valueRanges[j][1])
        if origines is not None:
            ax[j].plot(origines[1], origines[0], 'r*', markersize=3)
        if myoOpenings is not None:
            ax[j].plot(myoOpenings[:,1], myoOpenings[:,0], 'c^', markersize=3)
        if endoCenters is not None:
            ax[j].plot(endoCenters[1,], endoCenters[0], 'r+', markersize=5)    
        ax[j].set(title=plotSubTitles[j])
    fig.suptitle(suptitle)
    return fig, ax
    

def plot_2d_data_contours_segments(data, labelColors=None, 
                                   contours=None, segments=None,  
                                   typeContShow='all', typeSegShow='all',
                                   cmap='gray', figsize=(15,10), figshape=None,
                                   vmin=None, vmax=None, 
                                   vminSeg=0, vmaxSeg=1,
                                   linewidth=1, markersize=1, 
                                   ax = None,
                                   suptitle='2D view contour & segments in slices',
                                   titles=None,
                                   xlabel='Dim 1', ylabel='Dim 2',
                                   roiScattered = False
                                   ):
    """ show 2d data image with segments 
    """
    if len(data.shape) == 2:
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=figsize)
        else:
            fig = None
        # show dcm data
        ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
        # show contours if input
        if contours is not None and typeContShow is not None:
            if labelColors.keys() != contours.keys():
                raise TypeError('keys in labelColors should be the same as in contours')
            if roiScattered:
                scatter_contours_inSlice(ax, contours, labelColors, typeContShow, markersize)
            else:
                show_contour_inSlice(ax, contours, labelColors, typeContShow, linewidth)
        # show segments if input    
        if segments is not None and typeSegShow is not None:
            if labelColors.keys() != segments.keys():
                raise TypeError('keys in labelColors should be the same as in segments')  
            if roiScattered:
                scatter_segments_inSlice(ax, segments, labelColors, typeSegShow, markersize)
            else:
                show_segments_inSlice(ax, segments, labelColors, typeSegShow,
                                      vmin=vminSeg, vmax=vmaxSeg)
    else:          
        numSlices = data.shape[-1]
        if ax is None:
            if figshape is None:
                figNumCol = np.ceil(np.sqrt(numSlices)).astype(int)
                figNumRow = np.ceil(numSlices/figNumCol).astype(int)
            else:
                figNumRow,figNumCol = figshape
            fig, ax = plt.subplots(figNumRow, figNumCol, figsize=figsize)
            ax = np.array(ax)
        else:
            fig = None
        # show dcm data
        for i in range(0, numSlices):
            if not np.all( np.isnan(data[:,:,i])):
                ax[np.unravel_index(i, ax.shape)].imshow(data[:,:,i], cmap=cmap, vmin=vmin, vmax=vmax)
            else: # show white cavas if no image content
                ax[np.unravel_index(i, ax.shape)].imshow(np.full(np.shape(data[:,:,i]),fill_value=0), cmap='binary')
           
        # show contours if input
        if contours is not None and typeContShow is not None:
            if len(contours) != numSlices:
                raise TypeError('slices number in data and contours are not the same.')
            if not all(elem in labelColors.keys() for elem in contours[0].keys()):
                raise TypeError('keys in contours should be in labelColors')
            if roiScattered:
                for i in range(0, numSlices):
                    scatter_contours_inSlice(ax[np.unravel_index(i, ax.shape)], 
                                          contours[i], labelColors, typeContShow, markersize)
            else:
                for i in range(0, numSlices):
                    show_contour_inSlice(ax[np.unravel_index(i, ax.shape)], 
                                         contours[i], labelColors, typeContShow, linewidth)
                    
        # show segments if input    
        if segments is not None and typeSegShow is not None:
            if len(segments) != numSlices:
                raise TypeError('slices number in data and segments are not the same.')    
            if not all(elem in labelColors.keys() for elem in segments[0].keys()):
                raise TypeError('keys in segments should be in labelColors')  
            if roiScattered:
                for i in range(0, numSlices):
                    scatter_segments_inSlice(ax[np.unravel_index(i, ax.shape)], 
                                             segments[i], labelColors, typeSegShow, markersize)
            else:
                for i in range(0, numSlices):
                    show_segments_inSlice(ax[np.unravel_index(i, ax.shape)], 
                                          segments[i], labelColors, typeSegShow,
                                          vmin=vminSeg, vmax=vmaxSeg)
                    
        if titles is not None:
            if len(titles) != numSlices:
                raise ValueError('titles should be a list (of title strings) of the same length as slices number')
            for i in range(0, numSlices):
                ax[np.unravel_index(i, ax.shape)].set_title(titles[i], size=10)
                
        if numSlices < ax.size:
            for i in range(numSlices, ax.size):
                ax[np.unravel_index(i, ax.shape)].set_visible(False)
                
    # ax_share = fig.add_subplot(111, frameon=False)
    # ax_share.set(xlabel=xlabel, ylabel=ylabel, xticklabels=[], yticklabels=[])
    if fig is not None:
        fig.text(0.5, 0.01, xlabel, ha='center')
        fig.text(0.01, 0.5, ylabel, va='center', rotation='vertical')
        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=15)  
        plt.tight_layout()     
    return fig, ax


def plot_3d_contours(contours, USliceLocation, labelColors, typeContShow = 'all',
                     figsize=(8,6), suptitle='3D view contours in slices'):
    """ plot contours in slices in a 3d figure
        Paramters: 
            contours: contour attribute of mi.common.funObj.RoiObj
                list of contours info of each slice
            figsize: tuple, size of figure
    """
    if len(contours) != len(USliceLocation):
        raise ValueError('lenth of contours and USliceLocation should be the same.')
    if contours[0].keys() != labelColors.keys():
        raise ValueError('The keys in entris of contours are not the same as in labelColors.')
    fig_3d = plt.figure(figsize=figsize)
    fig_3d.suptitle(suptitle) 
    ax_3d = fig_3d.add_subplot(projection='3d')
    if typeContShow == 'all':
        typeContShow = contours[0].keys()
    for i in range(0, len(contours)):
        for key in typeContShow:
            val = contours[i][key]
            if len(val) > 0:
                if key == 'non-MI':
                    for j in range(len(val)):
                        ax_3d.plot(val[j][:,0], val[j][:,1],
                                    zs=USliceLocation[i], 
                                    linewidth = 0.8,
                                    c=labelColors[key][j])
                else:
                    for j in range(len(val)):
                        ax_3d.plot(val[j][:,0], val[j][:,1],
                                    zs=USliceLocation[i], 
                                    linewidth = 0.8,
                                    c=labelColors[key])
    ax_3d.azim = 20; ax_3d.elev = 20
    ax_3d.set(xlabel='Dim 1', ylabel='Dim 2', zlabel='Slice location') 
    if np.diff(USliceLocation).mean() < 0:
        ax_3d.invert_zaxis()
    plt.tight_layout()
    return fig_3d, ax_3d

def plot_3d_normalized_surface(data, contours, z, segments, labelColors, 
                               figsize=(8,6), suptitle='3D of normalized data'):
    if not all(elem in labelColors.keys() for elem in segments[0].keys()):
                raise ValueError('keys in segments should be in labelColors')
    if len(segments) != data.shape[-1]:
        raise ValueError('lenth of segments should be equal to slice number.')
    H,W,Z = [],[],[]
    for i in range(len(z)):
        H += contours[i]['non-MI'][0][:,0].tolist()
        W += contours[i]['non-MI'][0][:,1].tolist()
        Z += [z[i]]*contours[i]['non-MI'][0].shape[0]
    # W = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]),
    #                    indexing='ij')
    fig_3d = plt.figure(figsize=figsize)
    fig_3d.suptitle(suptitle) 
    ax_3d = fig_3d.add_subplot(projection='3d')
    ax_3d.plot_trisurf(W, H, Z, linewidth=0.2, antialiased=True)
    # ax_3d.plot_surface(W,H,Z,facecolors=colors, linewidth=0)
    return fig_3d, ax_3d


def plot_chamber(chamberView, figsize=(5,5),  cmap='gray', title='chamber view', axisoff=True):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(chamberView, cmap=cmap,
               interpolation='nearest', aspect='auto')
    if axisoff:
        ax.set_axis_off()
    fig.suptitle(title)
    return fig, ax


def scatter_contours_inSlice(ax, contours, labelColors, typeContShow='all', markersize=1):
    """ scatter contours within 1 slice
    """
    if typeContShow == 'all':
        typeContShow = list(contours.keys()) 
    dataColorGroup = funImg.groupup_dataColor({key: contours[key] for key in typeContShow}, 
                                              {key: labelColors[key] for key in typeContShow})
    for key in typeContShow:
        if len(contours[key]) > 0:
            ax.scatter(dataColorGroup[key]['data'][:,1],
                       dataColorGroup[key]['data'][:,0],
                       c=dataColorGroup[key]['colors'], 
                       s=markersize,
                       alpha=1)
    return 

def show_contour_inSlice(ax, contours, labelColors, typeContShow='all', linewidth=1):
    """ show contour line within 1 slice
    """
    if typeContShow == 'all':
        typeContShow = list(contours.keys()) 
    for key in typeContShow:
        val = contours[key]
        if len(val) > 0:
            if key == 'non-MI':
                for j in range(len(val)):
                    ax.plot(val[j][:,1], val[j][:,0],
                            linewidth = linewidth,
                            c=labelColors[key][j])
            else:
                for j in range(len(val)):
                    ax.plot(val[j][:,1], val[j][:,0],
                            linewidth = linewidth,
                            c=labelColors[key])        
    return

def scatter_segments_inSlice(ax, segments, labelColors, typeSegShow='all', markersize=1):
    """ scatter segments within 1 slice
    """
    if typeSegShow == 'all':
        typeSegShow = list(segments.keys()) 
    seg_locs = {key: funArray.mask_to_index(val) for key, val in segments.items()}
    seg_vals = {key: val[ np.atleast_2d(seg_locs[key])[:,0], np.atleast_2d(seg_locs[key])[:,1] ] \
                if seg_locs[key].size>0 else np.array([]) for key, val in segments.items()}
    dataColorGroup = funImg.groupup_dataColor({key: seg_locs[key] for key in typeSegShow}, 
                                              {key: labelColors[key] for key in typeSegShow},
                                              {key: seg_vals[key] for key in typeSegShow})
    for key in typeSegShow:
        if len(seg_locs[key]) > 0:
            ax.scatter(dataColorGroup[key]['data'][:,1],
                       dataColorGroup[key]['data'][:,0],
                       c=dataColorGroup[key]['colors'], 
                       s=markersize,
                       alpha=1)
    return

def show_segments_inSlice(ax, segments, labelColors, typeSegShow='all', vmin=None, vmax=None):
    """ show segments within 1 slice
        Paramters:
            ax: axes.Axes or array of Axes
            segments: dict contains different segments 
            labelColors: dict, segments corresponding colors
            typeSegShow: list of type (string) of segments to show
                'all' show all type of segments
            vmin, vmax: if not None 
    """
    if typeSegShow == 'all':
        typeSegShow = list(segments.keys()) 
        
    # colors = [[0,0,0]]; i = 1
    # canvas = np.zeros(list(segments.values())[0].shape)                
    # for key in typeSegShow:
    #     if np.sum(segments[key]) > 0:
    #         colors.append(labelColors[key][0] if isinstance(labelColors[key][0], list) 
    #                       else labelColors[key])
    #         canvas[segments[key]==1] = i
    #         i += 1
    # cmap = clrs.ListedColormap(colors, name='segmentColors', N=None)
    # ax.imshow(canvas, cmap=cmap)
    
    canvas = np.zeros(list(segments.values())[0].shape + (4,)) # RGBA
    for key in typeSegShow:
        if np.nansum(segments[key]) > 0:
            color = labelColors[key][0] if isinstance(labelColors[key][0], list) else labelColors[key]
            locs = segments[key] > 0
            alpha = segments[key][locs]
            if vmin is not None and vmax is not None:
                alpha[alpha>vmax] = vmax; alpha[alpha<vmin] = vmin
            else:
                if np.max(alpha) - np.min(alpha) > 0:
                    alpha = (alpha - np.min(alpha))/(np.max(alpha) - np.min(alpha))
            canvas[locs, :] = np.hstack((np.tile(color, (locs.sum(),1)), 
                                         alpha[:,np.newaxis]))
    ax.imshow(canvas, vmin=0, vmax=1)
    return 

def plot_coord_rad(ax, idx_myo, rad):
    """ plot radius coordinates
        Paramters:
            ax: axes.Axes or array of Axes
            idx_myo: [N,2] location of myocardium roi
            rad: [N,] rad value of idx_myo
    """
    
    return 

def tellme(s, draw=True):
    print(s)
    plt.title(s, fontsize=10)
    if draw:
        plt.draw()
    return

def show_17seg_bullsEye(ax, separations, 
                        color = 'gray', linestyle = 'solid' , linewidth = 1):
    for line in separations:
        ax.plot(line[:,1], line[:,0],
                color = color, linestyle = linestyle, linewidth = linewidth)
    return


def show_17seg_lineSeparations_inSlice(ax, lineSeparations, 
                                       color = 'gray', linestyle = 'solid', 
                                       linewidth = 1):
    if len(lineSeparations) > 0:
        for line in lineSeparations:
        # for line in lineSeparations:
            ax.plot(line[:,1], line[:,0],
                    color = color, linestyle = linestyle, linewidth = linewidth)
    return ax

def show_origine(ax, origines, color = 'red', marker = '.', markersize = 5):
    if len(origines.shape) > 1:
        for i in range(origines.shape[-1]):
            ax[np.unravel_index(i, ax.shape)].plot(origines[1,i], origines[0,i],
                                                   color = color, marker = marker, markersize = markersize)
    else:
        ax.plot(origines[1], origines[0], color = color, marker = marker, markersize = markersize)
    return ax

def show_endoCenter(ax, endoCenters, color = 'red', marker = '+', markersize = 3):
    if len(endoCenters.shape) > 1:
        for i in range(endoCenters.shape[-1]):
            ax[np.unravel_index(i, ax.shape)].plot(endoCenters[1,i], endoCenters[0,i], 
                                                   color = color, marker = marker, markersize = markersize)
    else:
        ax.plot(endoCenters[1], endoCenters[0], color, marker = marker, markersize = markersize)
    return ax
    
def show_myoOpening(ax, myoOpenings, color = 'cyan', marker = '^', markersize = 3):
    if len(myoOpenings.shape) > 1:
        for i in range(myoOpenings.shape[-1]):
            ax[np.unravel_index(i, ax.shape)].plot(myoOpenings[:,1,i], myoOpenings[:,0,i], 
                                                   color = color, marker = marker, markersize = markersize,
                                                   linestyle="None")
    else:
        ax.plot(myoOpenings[:,1], myoOpenings[:,0], 
                color = color, marker = marker, markersize = markersize,
                linestyle="None")
    return ax

def sequence_cmap(color, N=256):
    """ create sequential color map of self defined color 
        Paramters:
            color: list of RGB colors e.g.: [0.5,0.5,0.5] element value in [0,1]
            N: step
        Return: colormap
    """
    cmap = np.ones((N,4))
    cmap[:,:3] = np.array([np.linspace(1, color[i], N) for i in range(3)]).T
    cmap = ListedColormap(cmap)
    return cmap

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def showPlotlyFig(fig):
    import io
    import plotly.io as pio
    from PIL import Image
    buf = io.BytesIO()
    pio.write_image(fig, buf)
    img = Image.open(buf)
    img.show() 
    return






# def plot_2d_data_contours(data, contours, labelColors, 
#                           cmp='gray', figsize=(15,10), 
#                           suptitle='2D view contours in slices',
#                           xlabel='Dim 1', ylabel='Dim 2'):
#     """ show 2d data image with contours
#     """
#     if data.shape[-1] != len(contours):
#         raise TypeError('slices number in data and contours are not the same.')
#     numSlices = len(contours)
#     figNumCol = np.ceil(np.sqrt(numSlices)).astype(int)
#     figNumRow = np.ceil(numSlices/figNumCol).astype(int)
#     fig, ax = plt.subplots(figNumRow, figNumCol, figsize=figsize)
#     fig.suptitle(suptitle) 
#     for i in range(0, numSlices):
#         # show dcm
#         ax[np.unravel_index(i, ax.shape)].imshow(data[:,:,i], cmap='gray')
#         # show roi
#         scatter_contours_inSlice(ax[np.unravel_index(i, ax.shape)], 
#                                  contours[i], labelColors)
#     ax_share = fig.add_subplot(111, frameon=False)
#     ax_share.set(xlabel=xlabel, ylabel=ylabel)
#     plt.tight_layout() 
#     return fig, ax

