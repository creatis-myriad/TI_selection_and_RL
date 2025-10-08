#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:40:37 2020

@author: zheng
"""

import sys
sys.path.append('../../') 

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import shapely.geometry as geo
import shapely.ops as ops

import PolarLV.common.funArray as funArray 
# import polarLV.common.funPlot as funPlot

def tellme(s, draw=True):
    print(s)
    plt.title(s, fontsize=10)
    if draw:
        plt.draw()
    return

def crop_dcmRoi(dcm, roi, edge = 10):
    """ Crop image data and roi (contours) to focus on the roi
        Paramters: 
            dcm: mi.common.funObj.DicomObj object that contains image data
            roi: mi.common.funObj.RoiObj object that contains contours
            edge: float, [mm], the edge around the contour
        Returns: 
            dcm: dcm_cropped
            dcm: roi_cropped
    """
    if roi.points <= 0:
        raise ValueError('No contour found in slices, crop can not be processed.')
    contourstack = np.empty((0,2))
    limMax = np.asarray(np.shape(dcm.data)[0:2])-1
    limMin = np.zeros((2))
    for i in range(0, len(roi.contours)):
        if len(roi.contours[i]['non-MI'])>0: # myo as roi 
            contourstack = np.vstack((contourstack,
                                      np.vstack([cont for cont in roi.contours[i]['non-MI']])))
        # for key, val in roi.contours[i].items():
        #     if len(val)>0:
        #         tmp_contourstack = np.vstack([cont for cont in val])
        # contourstack = np.vstack([contourstack, tmp_contourstack])
    pixelEdge = edge//np.abs(np.asarray(dcm.spacing[0:2]))
    tmp_min = (np.min(contourstack, 0) - pixelEdge).astype(int)
    tmp_max = (np.max(contourstack, 0) + pixelEdge).astype(int)
    tmp_max[tmp_max>limMax] = limMax[tmp_max>limMax] 
    tmp_min[tmp_min<limMin] = limMin[tmp_min<limMin] 
    
    dcm.data = dcm.data[tmp_min[0]:tmp_max[0]+1,
                        tmp_min[1]:tmp_max[1]+1,
                        :]
    dcm.offset[0:2] = [tmp_min[0], tmp_min[1]]
    for i in range(0, np.shape(dcm.data)[-1]):
        for key, val in roi.contours[i].items():
            roi.contours[i][key] = [cont-tmp_min for cont in val]   
            if len(roi.segments[i][key])>0:
                roi.segments[i][key] = roi.segments[i][key][tmp_min[0]:tmp_max[0]+1,
                                                            tmp_min[1]:tmp_max[1]+1] 
    return dcm, roi

def groupup_dataColor(data, colors, alpha=None):
    """ Group up the data and color for plot, scatter etc
        Paramters:
            data: dictionary 
                data[key] can be list, with all entries from the same group or
                each entry from a group.
                The row (1st dim) of data[key] should represent a sample
            color: dictionary with the same keys as data
                if several different color for the same key,
                the color should be arranged in rows 
                e.g. [[0,1,0],
                      [1,0,1]]
                and the color (row) numbers should be the same as len(data[key])
            alpha: dictionary
                each element represents an alpha value corresponds to a row (location) in data[key]
        Returns:
            grouped_datacolor
    """
    if isinstance(data, dict) and isinstance(colors, dict):
        if data.keys() != colors.keys():
            raise TypeError('data keys and colors keys should be the same.')
        grouped_datacolor = {key: {'data':[], 'colors':[]} for key in colors}
        for key, val in data.items():
            if len(val) > 0:
                datastack = np.vstack([cont for cont in val])
                if alpha is not None:
                    alphastack = np.hstack([cont for cont in alpha[key]])
                    if np.max(alphastack) - np.min(alphastack) > 0:
                        alphastack = (alphastack - np.min(alphastack)) / \
                            (np.max(alphastack) - np.min(alphastack))
                else:
                    alphastack = np.ones(datastack.shape[0])
                if isinstance(colors[key][0], list):
                    if len(val) == len(colors[key]):
                        colorstack_list = [np.tile(colors[key][i], (val[i].shape[0],1)) \
                                           for i in range(len(colors[key]))]
                        colorstack = np.vstack(colorstack_list)
                    elif len(val) < len(colors[key]):
                        colorstack = np.tile(colors[key][0], 
                                             (np.shape(datastack)[0],1))
                    else:
                        raise TypeError('number of arrays in data is greater than' + \
                                        ' the number of colors, key {:s}'.format(key))
                else:
                    colorstack = np.tile(colors[key], 
                                         (datastack.shape[0],1))
                grouped_datacolor[key]['data']   = datastack            
                grouped_datacolor[key]['colors'] = np.hstack((colorstack, 
                                                              alphastack[:,np.newaxis]))
    else:
        raise TypeError('The function works only for dicts currently.')                        
    return grouped_datacolor


def purify_contours(contours, hierarchy):
    """ remove the contours that is not a circle
        - special case: if the array is almost a line but only its 4 points form a rhombus without pixel inside 
            we'll consider this case as a mistake, and this contour will be suppremed 
        Paramters: 
            contours: list of contours returned by cv2.findContours
            hierarchy: list of hierarchy returned by cv2.findContours
        Returns:
            contours: purified contours 
            hierarchy: purified hierarchy
    """
    idx_circle = []
    for i in range(len(contours)):
        cont = np.atleast_2d(contours[i].squeeze())
        if funArray.isPolygon(cont):
            cont_poly = geo.Polygon([tuple(cont[i]) for i in range(cont.shape[0])])
            if cont_poly.area <= 1 and cont.shape[0] > 10: # special case
                pass 
            else:
                idx_circle.append(i)
    contours = [contours[i] for i in idx_circle]
    hierarchy = [np.take(hierarchy[0], idx_circle, axis=0)]
    return contours, hierarchy

def findSegments_fromContours(contours, canvasShape):
    """ get segments from contours (in 1 slice) 
        Paramters:
            contours: dict contains contours of different type
                contours dimension [Height, Width] to be consist with imshow
            canvasShape: tuple, the shape of canvas 
        Returns:
            segments: dict contains segments according to contours
    """
    segments = {key: [] for key in contours}
    for key, val in contours.items():
        canvas = np.zeros(canvasShape)
        if len(val) > 0: # contour found
            cv.fillPoly(canvas, [np.fliplr(cont).astype(np.int32) for cont in val], color=1)
            # if key == 'non-MI': # myocardium
            #       if len(contours)<2:
            #           cv.fillPoly(canvas, [np.fliplr(val[0])], color=1)
            #       else:
            #           cv.fillPoly(canvas, [np.fliplr(val[0])], color=1)
            #           cv.fillPoly(canvas, [np.fliplr(val[1])], color=0)
            # else:
            #     cv.fillPoly(canvas, [np.fliplr(cont) for cont in contours[key]], color=1)
        segments[key] = canvas.astype(int)
    return segments

def findContours_fromSegments(segments):
    """ get contours from segments (in 1 slice)
        Paramters: 
            segments: dit contains segments of different type
        Returns:
            contours: dict contains contours according to the segments
                contours dimension [Height, Width] to be consist with imshow
    """
    contours = {key: [] for key in segments}
    for key, val in segments.items():
        contours_tmp, hierarchy = cv.findContours(val.astype(np.uint8), 
                                                  cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        if len(contours_tmp) > 0: # contour found
            contours_tmp = [np.fliplr(np.atleast_2d(cont.squeeze())) \
                            for cont in contours_tmp] # do exchange of axis make contours [H, W]
            if key == 'non-MI': # myocardium
                if hierarchy[0].shape[0]>2:
                    raise ValueError('More than 2 contours found for "non-MI" ?')
                elif hierarchy[0].shape[0]<2:
                    contours[key] = contours_tmp
                else:
                    if hierarchy[0][0,2]==1: # contours[0] is epi
                        contours[key] = contours_tmp
                    else:
                        contours_tmp.reverse()
                        contours[key] = contours_tmp
            else: # key in ['MI', 'NR']
                contours[key] = contours_tmp       
    return contours

def find_endoCenter(endo):
    """ find the endo center location (in 1 slice)
        Paramters: 
            endo: [N,2] with each row denotes location [Height, Width]
                must be a polygon
        Returns:
            endoCenter: [2,] the center of endo [Height, Width]
    """
    if len(endo) < 1:
        endoCenter = np.full((2), np.nan)
    else:
        if not funArray.isPolygon(endo):
            raise ValueError('endo should be a polygon!')
        canvas = np.zeros((int(np.max(endo[:,0])*2), int(np.max(endo[:,1])*2))) # no matter the size, just cover the endo
        cv.fillPoly(canvas, [np.fliplr(endo).astype(np.int32)], color=1) 
        endoCenter = np.mean(np.array(np.where(canvas==1)), axis=1).astype(int)
    return endoCenter
    
def locate_origine_myoOpening(fig):
    """ interact with figure to select origine and myocadium opening
        Paramters:
            fig: figure of data and myo contours and segments info
        Returns:
            origine: np array [2,] x,y location of picked origine
            myoOpening: np array [2,2] locations of myoOpenings 
                myoOpening[0,:] is the point closer to origine
    """
    while True:
        pts = []; tellme('Draw 1 point RV-LV (+ 2 points myo opening if exist)')
        pts = fig.ginput(4, timeout=-1, show_clicks=True)
        if len(pts) < 1:
            tellme('no point selected, key click for yes, mouse click for redo')
            if plt.waitforbuttonpress():
                break
        if len(pts) == 2:
            tellme('only 1 point selected for myo opening, redo it')
        if len(pts) > 3:
            tellme('{:d} point a selected (3 max), redo it'.format(len(pts)))
        if len(pts) == 1 or len(pts) == 3:
            ps, = plt.plot([pt[0] for pt in pts], [pt[1] for pt in pts], 
                           'r+', markersize=9) # plot the selected points
            tellme('Happy? key click for yes, mouse click for redo')
            if plt.waitforbuttonpress():
                break
            else:
                ps.remove()
        continue
    origine    = list(pts[0]) if len(pts) > 0 else np.full((2), np.nan)
    myoOpening = np.array(pts[1:3]) if len(pts) > 2 else np.full((2,2), np.nan)
    origine    = np.flip(origine) # exchange dimension to [H, W]
    myoOpening = np.fliplr(myoOpening)
    # make sure the first myoOpening point is the one closer to origine
    if np.linalg.norm(myoOpening[0,:]-origine) > np.linalg.norm(myoOpening[1,:]-origine):
        myoOpening = np.flipud(myoOpening)
    return origine, myoOpening

def interact_draw_line(fig, ax):
    """ interact with figure to pick 2 points to draw a line
        Paramters: 
            fig: figure
            ax:  axes.Axes or array of Axes
        Returns:
            pts: np array [2,2], the 2 end points 
                pts[0,:] is closer to the natural origine [0,0] 
    """
    while True:
        print('\n')
        tellme('Select 2 points to draw a RV to LV cut line, '
                       'which serves the estimation of the chamber view.')
        print('\n Ideally these 2 points locate at the centers of RV and LV')
        pts = plt.ginput(3, timeout=-1, show_clicks=True)
        if len(pts) == 2:
            pts = np.array(pts)
            if (pts[0,:] == pts[1,:]).all():
                tellme('The selected 2 points should not be the same')
                continue
            ax.plot(pts[:,0], pts[:,1], color='orange', marker='*')
            tellme('\ndrawed line in orange')
            tellme('Happy? key click for yes, mouse click for redo')
            if plt.waitforbuttonpress():
                print('\n'); break
        else:
            tellme('2 points are needed')
        continue
        if pts[1,0] < pts[0,0]:
            pts = np.flipud(pts)
    return pts

def extend_line_to_edge(lineEnds, canvasSize, ax=None):
    """ extend the two ends of lines, NEED IMPROVEMENT !!!!!
        Paramters: 
            lineEnds: [2,2], lineEnds[0,:] first end point
            canvasSize: the size of canvas: tuple or list 
        Returns:
            lineEnds_edge: the lineEnds touches the edge of canvas
    """
    # make sure first end point is closer to natural origine
    lineEnds_edge = np.zeros((2,2))
    lineEnds_edge[0,0], lineEnds_edge[0,1], \
    lineEnds_edge[1,0], lineEnds_edge[1,1] = extend_line(0, 0, 
                                                         canvasSize[0]-1, canvasSize[1]-1, 
                                                         lineEnds[0,0], lineEnds[0,1], 
                                                         lineEnds[1,0], lineEnds[1,1])
    if lineEnds_edge[1,0] < lineEnds_edge[0,0]:
        lineEnds = np.flipud(lineEnds) 
    if ax is not None:
        ax.plot(lineEnds_edge[:,0], lineEnds_edge[:,1], color='orange', marker='o')
        tellme('line is extended to the edge')
    return lineEnds_edge

def fill_line(lineEnds, canvasSize):
    """ fill the line (pixels within the two ends)
        if lineEnds are not integers, they are rounded
        Paramters: 
            lineEnds: [2,2] points of the two ends
                lineEnds[0,:] denote the location of an end
        Returns:
            line: filled 
    """
    canvas = np.zeros(canvasSize)
    cv.line(canvas, tuple(np.round(lineEnds[0,:]).astype(int)), 
            tuple(np.round(lineEnds[1,:]).astype(int)), color=1)
    return canvas.astype(int)
 

def extend_line(xmin, ymin, xmax, ymax, x1, y1, x2, y2):
    """ extend the line to the edge of plane,
        xmin <= x1, x2 <= xmax
        ymin <= y1, y2 <= ymax
        Paramters:
            xmin, ymin: the edge min of the plane
            xmax, ymax: the edge max of the plane
            x1, y1: one end of the line in the plane
            x2, y2: the other end of the line in the plane
        Returns:
            x1_ext, y1_ext: one end of the extended line
            x2_ext, y2_ext: the other end of the extended line
    """
    if y1 == y2:
        return (xmin, y1, xmax, y1)
    if x1 == x2:
        return (x1, ymin, x1, ymax)
    else:
        a = (y2-y1)/(x2-x1) 
        b = y1 - a * x1
        y_xmin = a * xmin + b
        y_xmax = a * xmax + b
        x_ymin = (ymin - b)/a
        x_ymax = (ymax - b)/a
    pts = []
    if ymin <= y_xmin <= ymax:
        pts.append([xmin, y_xmin])
    if ymin <= y_xmax <= ymax:
        pts.append([xmax, y_xmax])
    if xmin <= x_ymin <= xmax:
        pts.append([x_ymin, ymin])
    if xmin <= x_ymax <= xmax:
        pts.append([x_ymax, ymax])
    if len(pts) != 2:
        raise ValueError('Sth wrong, find {:d} points as extension'.format(len(pts)))
    if np.linalg.norm(np.array(pts[0])-np.array([x1,y1])) > np.linalg.norm(np.array(pts[0])-np.array([x2,y2])):
        pts.reverse()
    x1_ext, y1_ext = tuple(pts[0])
    x2_ext, y2_ext = tuple(pts[1])
    return x1_ext, y1_ext, x2_ext, y2_ext

def interpolate_2d_images(img, samplingFactor, mtdInterp='linear'):
    """ interpolate 2d images
        can be done by scipy.ndimage.zoom also.
        e.g. img = [[1,3,5],
                    [2,4,6]]
            samplingFactor = 2, 
            imgResampled, somehow [[1,   1.5,  3,   4.5, 5],
                                   [1.5, 2,    3.5, 4,   5.5],
                                   [2,   3     4,   5,   6]]  
        Paramters: 
            img: ndarray, a sigle image data [n_x, n_y] or 
                a pile of image data [n_x, n_y, numberSlices]
            samplingFactor: upsampling factor 
            mtdInterp: iterpolation method, see scipy.interpolate.griddata
        Returns:
            imgResampled: interpolated image
    """
    if len(img.shape) == 2: 
        n_x, n_y = img.shape
        imgResampled = np.zeros(((n_x-1)*samplingFactor+1, 
                                 (n_y-1)*samplingFactor+1))
        grid_x, grid_y = np.mgrid[0:n_x-1:n_x*1j, 0:n_y-1:n_y*1j]
        grid_x_interp, grid_y_interp = np.mgrid[0:n_x-1:((n_x-1)*samplingFactor+1)*1j, 
                                                0:n_y-1:((n_y-1)*samplingFactor+1)*1j]
        imgResampled = interpolate.griddata(np.vstack((grid_x.ravel(), grid_y.ravel())).T, 
                                            img.ravel(), 
                                            (grid_x_interp, grid_y_interp),
                                            method = mtdInterp) 
    elif len(img.shape) == 3:
        n_x, n_y, numSlices = img.shape
        imgResampled = np.zeros(((n_x-1)*samplingFactor+1, 
                                 (n_y-1)*samplingFactor+1, 
                                 numSlices))
        grid_x, grid_y = np.mgrid[0:n_x-1:n_x*1j, 0:n_y-1:n_y*1j]
        grid_x_interp, grid_y_interp = np.mgrid[0:n_x-1:((n_x-1)*samplingFactor+1)*1j, 
                                                0:n_y-1:((n_y-1)*samplingFactor+1)*1j]
        for i in range(0, numSlices):
            imgResampled[:,:,i] = interpolate.griddata(np.vstack((grid_x.ravel(), grid_y.ravel())).T, 
                                                       img[:,:,i].ravel(), 
                                                       (grid_x_interp, grid_y_interp),
                                                       method = mtdInterp)
    else:
        raise TypeError('img dimension error, input image should be either a sigle image or a pile of images')
    return imgResampled

def interpolate_segment(segment, samplingFactor, mtdInterp=None):
    """ interpolate segment 
        Paramters:
            contour: ndarray [N,2]
            samplingFactor: upsampling factor 
            mtdInterp: None will be linearly interpolated
        Returns:
            contourResampled: resampled contour
    """
    n_x, n_y = segment.shape
    segmentResampled = np.zeros(((n_x-1)*samplingFactor+1, 
                                 (n_y-1)*samplingFactor+1))
    contours, hierarchy = cv.findContours(segment.astype(np.uint8), 
                                         cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if len(contours) > 0: # contour found
        contours = [np.atleast_2d(cont.squeeze())*samplingFactor \
                    for cont in contours]  # do exchange of axis make contours [H, W]
    cv.fillPoly(segmentResampled, contours, color=1)
    if mtdInterp is not None:
        raise NotImplementedError('the mtdInterp is not implemented yet')
    return segmentResampled

def extend_line_to_edge_oneside(lineStart, lineEnd, canvasShape):
    """ get the intersect of line with the edge of canvas
        Paramters:
            lineStart: ndarray [2,] [H,W] start of the line
            lineEnd: ndarray [2,] [H,W] end of the line
            canvasShape: tuple, shape of the canvas
        Returns:
            lineExtend_edge: the intersect of the extended line with an adge of canvas
    """
    intersects = extend_line_to_edge(np.vstack((lineStart, lineEnd)), 
                                    canvasSize=canvasShape)
    # the intersect should be the one with the distance to lineStart greater than to lineEnd
    idx = (np.linalg.norm(intersects - lineStart, axis=1) - \
           np.linalg.norm(intersects - lineEnd, axis=1)) > 0
    lineExtend_edge = intersects[idx, :].squeeze()
    return lineExtend_edge

def find_roni(endoCenter, myoOpening, canvasShape):
    """ find region of no interest
        Paramters:
            endoCenter: ndarray, [2,] [H,W]
            myoOpening: ndarray, [2,2]
            canvasShape: tuple, the shape of canvas
        Returns:
            roni: region of no interest, 
                ndarray [4,] a polygon describes the region.
                if any value in endoCenter or myoOpening is nan, 
                roni = None
    """
    if np.any(np.isnan(np.vstack((endoCenter, myoOpening)))):
        roni = None
    else:
        roni = np.full((4, 2), fill_value=np.nan)
        roni[0,:] = endoCenter
        # the intersects of origine -- myoOpening extensions with the edge of canvas
        roni[1,:] = extend_line_to_edge_oneside(endoCenter, myoOpening[0,:], canvasShape)
        roni[3,:] = extend_line_to_edge_oneside(endoCenter, myoOpening[1,:], canvasShape)
        mean_intersects = (roni[1,:] + roni[3,:])/2
        if (mean_intersects[0] in [0, canvasShape[0]-1]) or \
            (mean_intersects[1] in [0, canvasShape[1]-1]):
            # take the mean of myoOpening as the 4th vertex if it already lies on the edge of canvas
            roni[2,:] = mean_intersects 
        else:
            # take the vertex of canvas which is the closest to the intersection 
            #   of canvas edge and the extended line of endoCenter - myoOpenings' mean 
            extended_mean_intersect =  extend_line_to_edge_oneside(endoCenter, 
                                                                   mean_intersects, 
                                                                   canvasShape)
            candidates = np.array([[0,0],
                                   [0,canvasShape[1]-1],
                                   [canvasShape[0]-1,0],
                                   [canvasShape[0]-1, canvasShape[1]-1]])
            dists = np.linalg.norm(candidates-extended_mean_intersect , axis=1)
            roni[2,:] = candidates[np.where(dists == np.min(dists))[0][0], :].squeeze()
    return roni

def shortestDist_point_to_contour(point, contour):
    """ compute the shortest distance from the point to a point on contour
            contour may not be a closed circle
        Paramters:           
            points: [2,] ndarray
            contour: [N,2] ndarray
        Returns:
            shortestDist: shortest distance
    """
    shortestDist = np.min(np.linalg.norm(contour - point, axis=1))
    return shortestDist

def separate_epi_endo(contour, endoCenter=None):
    """ separate epi & endo of myocardium when 
            they're in one contour present in ccrescent shape
        Paramters:
            contour: [N,2] ndarray of contour
            endoCenter: [2,] ndarray if provided
        Returns:
            epi: [M,2] ndarray of epi
            endo: [N-M,2] ndarray of endo
    """
    if endoCenter is None:
        endoCenter = find_endoCenter(contour)
    # Depend de la version de opencv : (4.5.1.48)==>tuple(endoCenters)
    #                                  (>= 4.5  )==>tuple(np.float32(funImg.find_endoCenter(contour_myo[0]))
    if cv.pointPolygonTest(np.float32(contour), tuple(np.float32(endoCenter)), False) >= 0:
        raise ValueError('the endoCenter is inside/on the contour, ' +\
                          'seems the contour is just an epi without endo?')
    # convert contour to linestring object
    line_contour = geo.LineString([tuple(contour[i]) \
                                   for i in range(contour.shape[0])])  
    convexHull = np.array(line_contour.convex_hull.boundary.coords.xy).T  # get the covex hull
    # edges = np.linalg.norm(np.diff(convexHull, axis=0), axis=1) # get the edges of the convex hull contour
    # idx_compenEdge = np.where(edges == np.max(edges))[0][0] # get the index of the max edge 
    # the 2 points forms the max edge very possible to be the join of endo & epi (not very robust)
    midPoint_toContour = [] # consider the mid of vertices of convexhull farest to line_contour
    for i in range(convexHull.shape[0]-1):
        midPoint_toContour.append(line_contour.distance(geo.point.Point((convexHull[i,:]+convexHull[i+1,:])/2)))
    idx_compenEdge = np.where(midPoint_toContour == np.max(midPoint_toContour))[0][0]
    joints_endoEpi = convexHull[idx_compenEdge:idx_compenEdge+2, :] 
    joints_endoEpi = np.array([funArray.closest_pointInContour_to_point(contour, 
                                                                        joints_endoEpi[0,:])[0],
                               funArray.closest_pointInContour_to_point(contour, 
                                                                        joints_endoEpi[1,:])[0]])
    
    line_split = ops.split(line_contour, 
                           geo.point.Point(tuple(joints_endoEpi[0,:])))
    if len(line_split) < 2:
        line_contour_beginWith1stJoin = line_split[0]
    else:
        line_contour_beginWith1stJoin = np.vstack((np.array(line_split[1]), 
                                                   np.array(line_split[0])[0:-1,:]))
        line_contour_beginWith1stJoin = geo.LineString( 
            [tuple(line_contour_beginWith1stJoin[i]) for i in range(contour.shape[0])])  
    line_split = ops.split(line_contour_beginWith1stJoin, 
                           geo.point.Point(tuple(joints_endoEpi[1,:])))
    dists = [line.distance(geo.point.Point(tuple(endoCenter))) \
             for line in line_split]
    candidates = [np.array(line_split[0]), np.array(line_split[1])[1:,:]]
    epi = candidates[np.where(dists == np.max(dists))[0][0]]
    endo = candidates[np.where(dists == np.min(dists))[0][0]]
    if epi.shape[0] + endo.shape[0] != contour.shape[0]:
        raise ValueError('Sth wrong, Epi + Endo lengh should equals to the entire contour length')
    return epi, endo

def rotate_img(image, angle, center=None, scale=1.0, borderValue=np.nan):
    """ rotate image around a specific center 
        Parameters:
            image: 2D ndarray
            angle: the angle of rotation (counterlockwise)
            center: specific center of the rotation, tuple or list or ndarray of location (h,w)
            scale: scale of rotation
            borderValue: the value that fills the rotation border
    """
    (h, w) = image.shape[:2]
    if center is None:
        center = (w/2, h/2)
    # Perform the rotation
    M = cv.getRotationMatrix2D(tuple(center), angle/np.pi*180, scale)
    rotated = cv.warpAffine(image, M, (w, h), borderValue=borderValue)
    return rotated




    




