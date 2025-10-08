#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:08:16 2020

@author: zheng
"""

# import numpy as np

import sys
sys.path.append('../../') 

import numpy as np
import warnings
import cv2 as cv
# import scipy.interpolate as interpolate
import shapely.geometry as geo
# import shapely.ops as ops

import PolarLV.common.funImg as funImg
import PolarLV.common.funCoord as funCoord
import PolarLV.math.basicStatics as bStats

# import polarLV.math.geometry as geo


class DataObj(object):
    pass

class DicomObj():
     def __init__(self, folder=None, 
                  UID=None, USliceLocation=None,
                  data=None, ImagePositionPatient=None,
                  offset=None, spacing=None):
         """ DICOM object
             Parameters: 
                 folder: str, the original folder which provides the data
                 UID: ndarray [n_slices,] of UID
                 USliceLocation: ndarray [n_slices,] of USliceLocations
                 data: ndarray [n_rows, n_columns, n_slices]
                 ImagePositionPatient: ndarray [3,]
                 offset: ndarray [3,], 3 offsets corresponding to the dims of data
                 spacing: ndarray [3,], 3 spacings corresponding to the dims of data
         """
         ## default
         self.folder = folder
         self.UID = UID
         self.USliceLocation = USliceLocation
         self.data = data
         self.ImagePositionPatient = ImagePositionPatient
         self.offset = offset
         self.spacing = spacing
         
         ## customized
         self.dataResampled = None
         self.spacingResampled = None
         
     def resample(self, samplingFactor, mtdInterp='linear'):
         """ resample data
             Parameters:
                 samplingFactor: int, the sampling factor 
                 mtdInterp: str, interpolation method
                     see scipy.interpolate.griddata for detail of the options
         """
         self.spacingResampled = list(np.array(self.spacing[0:2])/samplingFactor) + [self.spacing[-1]]
         self.dataResampled = funImg.interpolate_2d_images(self.data, 
                                                           samplingFactor, 
                                                           mtdInterp = mtdInterp)
         return
     
     def check_zSpacing(self, atol = 1e-06):
         """ check if z spacing corresponds to the diff of USliceLocations
         """
         USliceLocation_diff = np.diff(self.USliceLocation)
         if not np.all(np.isclose(USliceLocation_diff, self.spacing[-1], atol=atol)):
             print('\nNot all USliceLocation differences equal to registrated z spacing.')
             if np.all(np.isclose(USliceLocation_diff, USliceLocation_diff.mean(), atol=atol)):
                 print('Z spacing is adjusted (from {:3f}) '.format(self.spacing[-1]) + 
                       'to the real difference of USliceLocation ({:3f}).'.format(USliceLocation_diff.mean()))
                 self.spacing[-1] = USliceLocation_diff.mean()
             else:
                 print('Z spacing is not adjusted, sicne difference of USliceLocations are not the same.')
         return
     
     def inverse_slices(self):
         """ inverse slices, 
             update USliceLocation, UID, offset[-1] and spacing[-1] as well 
         """
         print('slices are inversed!')
         self.data = np.flip(self.data, -1)    
         self.UID  = np.flip(self.UID)   
         self.USliceLocation = np.flip(self.USliceLocation)  
         self.offset[-1]  = self.USliceLocation[0]
         self.spacing[-1] = -self.spacing[-1]
         return
     
     def get_data_with_coords(self, coordinates, resampled = True):
         """ Get resampled data that with valide coordinate values 
             Parameters:
                 coordinates: instance of roiObj.coordinates
             Returns:
                 data_withcoords: data which has it's values without valid coordinates set to nan.
         """
         if resampled:
             if self.dataResampled is None:
                 raise ValueError('The dataResampled should be calculated first using "resample()"')
             data = self.dataResampled
         else:
             data = self.data
         data_withcoords = funCoord.get_data_with_coords(data, coordinates)
         return data_withcoords
     
         
class RoiObj():
    def __init__(self, folder=None,
                 labelColors=None, points=None,
                 contours=None, segments=None):
        """ Region of interest object
            Parameters: 
                folder: str, the original folder which provides the contour info
                labelColors: dict with keys the contour types and values the corresponding colors
                points: int, the original denoted points number in contour file (not used)
                contours: list of dicts (a dict for a slice) in which,
                    each key denotes a contour type and corresponding value a list of contours (ndarray [n,2])
                    e.g. contours[0]['non-MI'][0], 1st contour of type 'non-MI' in the 1st slice
                segments: list of dicts (a dict for a slice) in which,
                    each key denotes a segment type (same as contour type) 
                    and corresponding value a ndarray (same size as dcm.data)
                    e.g. segments[0]['non-MI'], segment of type 'non-MI' in the 1st slice
        """
        ## default
        self.folder = folder
        self.points = points
        self.labelColors = labelColors
        self.contours = contours
        self.segments = segments
        
        ## customized
        self.origines    = None
        self.myoOpenings = None
        self.endoCenters = None
        self.idxes = {'apex': np.nan, 'valve': np.nan}

        self.segmentsResampled    = None
        self.contoursResampled    = None
        self.originesResampled    = None
        self.myoOpeningsResampled = None
        
        self.coordinates = None
    
    def add_segments(self, canvasShape, renew=False):
        """ Add segmentation according to contours
            Parameters:
                canvasShape: tuple (2,) the shape of canvas where segment will be drawn
                renew: bool, if True recompute the segments according to contours 
                    even it exists already
        """
        numSlices = len(self.contours)
        if numSlices < 1:
            raise TypeError('Input contours is empty.')
        if self.segments is not None and (not renew):
            pass
        else:
            self.segments = [{} for i in range(numSlices)]
            for i in range(numSlices):
                self.segments[i] = funImg.findSegments_fromContours(self.contours[i], canvasShape)
                
    def add_index_apex_valve(self, idx_apex, idx_valve):
        """ add index of apex and valve 
            Parameters: 
                idx_apex: int/float, the slice index of apex
                idx_value: int/float, the slice index of valve
        """
        if idx_valve <= idx_apex:
            raise ValueError('idx_valve can not be smaller than idx_apex.')
        self.idxes['apex']  = idx_apex
        self.idxes['valve'] = idx_valve
        return
    
    def add_endoCenters(self, mtdEndoCenter_mixEndoEpi='copy'):
        """ add endocadium center locations [H, W]
            which will take the mean locaion of the mask of endo
            Parameters:
                mtdEndoCenter_mixEndoEpi: str, the method to deal with the case
                    where endo & epi are not seperated, wich forms a crescent shape of myo.
                    'copy' will copy the endoCenter of the nearest slice 
                    'convexHull' will take the centroid of the convexHull of the crescent
                        as the endoCenter
        """
        if self.contours is None:
            raise AttributeError('contours are needed to compute endoCenters')
        self.endoCenters = np.full((2, len(self.contours)), np.nan) # apex to base
        id_fakeMyoOpen = []
        for i in range(len(self.contours)):
            if len(self.contours[i]['non-MI']) > 0:
                endoCenters = funImg.find_endoCenter(self.contours[i]['non-MI'][-1])
                # if the endoCenter is outside the contour
                # Depend de la version de opencv : (4.5.1.48)==>tuple(endoCenters)
                #                                  (>= 4.5  )==>tuple(np.float32(endoCenters))
                if cv.pointPolygonTest(self.contours[i]['non-MI'][-1].astype(np.int32), 
                                       tuple(np.float32(endoCenters)), False) >= 0:
                    self.endoCenters[:,i] = funImg.find_endoCenter(self.contours[i]['non-MI'][-1])
                else:
                    print('\n ! slice: {:d} seems have endo epi mixed'.format(i))
                    if mtdEndoCenter_mixEndoEpi == 'copy':
                        j = i # slice j-1 may be also without endocenter caused by e.g. no segments
                        while np.isnan(self.endoCenters[:,j-1]).any():
                            id_fakeMyoOpen.append(j); j = j-1 # find the previous slice with endocenter
                        self.endoCenters[:,i] = self.endoCenters[:,j-1]
                    elif mtdEndoCenter_mixEndoEpi == 'convexHull':
                        contour_tmp = self.contours[i]['non-MI']
                        if len(contour_tmp) >1:
                            raise ValueError('2 contours encontered when the contour should be a crescent shape')
                        line_contour = geo.LineString([tuple(contour_tmp[0][i]) \
                                                       for i in range(contour_tmp[0].shape[0])])  
                        # get the covex hull and compute the centroid
                        self.endoCenters[:,i] = np.array(line_contour.convex_hull.centroid.xy).squeeze()        
                    else:
                        raise NotImplementedError('Only copy and interpolate methods are interpolated.')
        # check if all slices in [apex, valve] has endo center
        for i in id_fakeMyoOpen:
            if mtdEndoCenter_mixEndoEpi == 'copy':
                self.endoCenters[:,i] = self.endoCenters[:,i+1]
            elif mtdEndoCenter_mixEndoEpi == 'convexHull':
                contour_tmp = self.contours[i]['non-MI']
                if len(contour_tmp) >1:
                    raise ValueError('2 contours encontered when the contour should be a crescent shape')
                line_contour = geo.LineString([tuple(contour_tmp[0][i]) \
                                               for i in range(contour_tmp[0].shape[0])])  
                # get the covex hull and compute the centroid
                self.endoCenters[:,i] = np.array(line_contour.convex_hull.centroid.xy).squeeze()    
            else:
                raise NotImplementedError('Only copy and interpolate methods are interpolated.')
        return
                
    def add_origines_myoOpenings(self, origines, myoOpenings, adjust=True):            
        """ Add origines and myoOpenings 
            Parameters: 
                origines: ndarray [2,n_slices], origines of slices
                myoOpenings: ndarray [2,2,n_slices], myoOpeing of slices
                    [0,:,0] represents the location of 1st myoOpening in 1st slice
                adjust: if True will adjust the origines and myoOpenings 
                    according to the indexes of apex and valve
        """
        if origines.shape[-1] != len(self.contours):
            raise TypeError('origines should have the same column number as contours length')
        if myoOpenings.shape[-1] != len(self.contours):
            raise TypeError('myoOpenings last dim should have the same length as contours length')
        if adjust:
            idx_hasOrigine = np.where(~np.isnan(origines[0,:]))[0]
            if self.idxes['apex'] is not None:
                apex_floor = np.floor(self.idxes['apex']).astype(int)
                valve_ceil = np.ceil(self.idxes['valve']).astype(int)
                if apex_floor < idx_hasOrigine[0]: # repeat the origine closest to apex until max(apex/0)
                    origines[:, max(0, apex_floor):idx_hasOrigine[0]] = \
                        origines[:,idx_hasOrigine[0],np.newaxis]
                elif apex_floor > idx_hasOrigine[0]: # reset the origines to nan below apex
                    origines[:, idx_hasOrigine[0]:apex_floor] = np.full((2,1), np.nan)
                else:
                    pass
                if valve_ceil > idx_hasOrigine[-1]: # repeat the origine closest to valve until min(valve/numSlices)
                    origines[:, (idx_hasOrigine[-1]+1): min(origines.shape[-1], valve_ceil+1)] = \
                        origines[:,idx_hasOrigine[-1],np.newaxis]
                elif valve_ceil < idx_hasOrigine[-1]: # reset the origines to nan above valve
                    origines[:, valve_ceil+1: idx_hasOrigine[-1]+1] = np.full((2,1), np.nan) 
                else:
                    pass
        self.origines = origines
        self.myoOpenings = myoOpenings
        return

    def resample(self, samplingFactor, mtdInterp=None):
        """ add resample of segments, contours, origines and myoOpenings 
            Parameters:
                samplingFactor: int
                mtdInterp: the method used when iterpolating for smoothing the contours
                    None by default, the contour will be interpolated linearly
        """
        if self.segments is None:
            raise AttributeError('attribute segments is needed for resampling, run roi.add_segments to get it')
        numSlices = len(self.segments)       
        self.segmentsResampled = [{key: [] for key in self.segments[0]} for i in range(numSlices)]
        self.contoursResampled = [{key: [] for key in self.segments[0]} for i in range(numSlices)]
        self.originesResampled    = np.full((2,numSlices), np.nan)  # apex to base
        self.myoOpeningsResampled = np.full((2, 2, numSlices), np.nan)  # apex to base
        for i in range(0, numSlices):
            for key, val in self.segments[i].items():
                self.segmentsResampled[i][key] = \
                    funImg.interpolate_segment(val, samplingFactor, mtdInterp=mtdInterp)
            self.contoursResampled[i] = funImg.findContours_fromSegments(self.segmentsResampled[i])
        if self.origines is not None:
            self.originesResampled = self.origines*samplingFactor
        if self.myoOpenings is not None:
            self.myoOpeningsResampled = self.myoOpenings*samplingFactor
        if self.endoCenters is not None:
            self.endoCentersResampled = self.endoCenters*samplingFactor
        return
    
    def add_coordinates(self, dcm, flag_debugPlot=False, figurePause=5):
        """ add polar coordinates
            Parameters:
                dcm: instance of funObj.DicomObj 
                flag_debugPlot: plot detected endo epi when they are mixed for debug
                figurePause: time wait before close the figure
        """
        self.coordinates = funCoord.compute_polar_coordinates(dcm, self, 
                                                              flag_plot = flag_debugPlot, 
                                                              figurePause = figurePause)        
        return
    
    def ravel_coordinates(self):
        """ ravel the coordinates 
            Returns: 
                coordinates_raveled: raveled coordinates [3, numCoordinates] ndarray
        """
        if self.coordinates is None:
            raise ValueError('coordinates is None, use "add_coordinates" to compute them first')
        coordinates_raveled = funCoord.ravel_coordinates(self.coordinates)
        return coordinates_raveled
    
    def get_z_vals(self):
        """ Get z values of each slices
            Returns:
                z_values: [numSlices,] ndarray of z value of each slice
        """
        if self.coordinates is None:
            raise ValueError('coordinates is None, use "add_coordinates" to compute them first')
        z_values = funCoord.get_z_vals(self.coordinates)
        return z_values
    
    def compute_missingAngles(self, minMissingSlot = 0.1):
        """ Compute missing angles caused by myoOpening
            Parameters:
                minMissingSlot: float, slot > minMissingSlot 
                    is considered as a missing angle caused by myoOpening
            Returns:
                missingAngles: [2, numSlices] ndarray beginings and ends of missing angle.
                    if no missing angle in the slice, [:, numSlice] should be [np.nan, np.nan]
        """
        radials = self.coordinates[:,:,1,:]
        numSlices = radials.shape[-1]
        missingAngles = np.full((2, numSlices), fill_value=np.nan)
        for i in range(numSlices):
            if ~np.all(np.isnan(radials[:,:,i])):
                radial_tmp = radials[:,:,i][~np.isnan(radials[:,:,i])]
                radial_tmp.sort()
                radial_tmp = np.append(radial_tmp, radial_tmp[0]+2*np.pi)
                diff_radial_tmp = np.diff(radial_tmp)
                if diff_radial_tmp.max() > minMissingSlot:
                    idx = np.where(diff_radial_tmp == diff_radial_tmp.max())[0][0]
                    missingAngles[:,i] = radial_tmp[idx:idx+2]
                    
        # deal with fake myo opening at middle slices
        id_withMissingAngles = np.where(~np.isnan(missingAngles[0,:]))[0]
        id_fakeMyoOpening = id_withMissingAngles[np.where(np.diff(id_withMissingAngles)>1)[0]]
        if len(id_fakeMyoOpening) > 0:
            warnings.warn('Slice {:s} seems has fake myoOpeining, it will be ignored\n'.format(str(id_fakeMyoOpening).strip('[]')))
            for i in id_fakeMyoOpening:
                missingAngles[:,i] = np.nan
             
        if self.myoOpeningsResampled is not None:
            id_withMissingAngles = np.where(~np.isnan(missingAngles[0,:]))[0]
            id_withMyoOpenings = np.where(~np.isnan(self.myoOpeningsResampled[0,0,0:np.ceil(self.idxes['valve']).astype(int)+1]))[0]
            if len(id_withMissingAngles) == len(id_withMyoOpenings):
                if not (id_withMissingAngles == id_withMyoOpenings).all():
                    warnings.warn('Slices with myo openings ({:s}) '.format(str(id_withMyoOpenings).strip('[]')) +
                                  'are not identical to slices with missing angles ({:s}). '.format(str(id_withMissingAngles).strip('[]')) +
                                  'MyoOpening annotations may be missing in some slices.')
            else:
                warnings.warn('Slices with myo openings ({:s}) '.format(str(id_withMyoOpenings).strip('[]')) +
                              'are not identical to slices with missing angles ({:s}).'.format(str(id_withMissingAngles).strip('[]')))
        return missingAngles
    
    # def get_valid_coordinate_vals(self):
    #     """ get valid coordinates values
    #         Return:
    #             radius_valid: [n_validRadVals,] ndarray, valid radius values
    #             angles_valid: [n_validAngVals,] valid angle values
    #             z_valid: [n_validZVals,] valid z values
    #     """
    #     if self.coordinates is None:
    #         raise ValueError('coordinates is None, use "add_coordinates" to compute them first')
    #     radius_valid, angles_valid, z_valid = funCoord.get_valid_coordinate_vals(self.coordinates)
    #     return radius_valid, angles_valid, z_valid
    
    
class RefObj():
    def __init__(self, maxZ=0.9, numSlices=21, shapeSlice=(80,80), 
                 offset=np.array([0,0,0]), spacing=np.array([0,0,5]),
                 resoRadial=200, maxRadEpi=50, maxRadEndo=30,
                 radEpiList=None, radEndoList=None,
                 origines='17seg',
                 mtdInterpImg = 'linear', mtdEndoCenter_mixEndoEpi = 'copy', 
                 mtdInterpContour = None, samplingFactor=4):
        """ left ventrical reference object
                from Apex to Valve (slice 0 = apex), 
                if radEpiList and radEndoList are not given, 
                the shape will be a tuncated oval defined by maxZ, maxRadEpi & maxRadEndo
            Parameters:
                maxZ: float < 1, to control the oval
                numSlice: int, slice number 
                shapeSlice: tuple (height, width) of each slice of dcm data
                offset: ndarray [3,] offsets
                spacing: ndarray, [3,] spacing 
                resoRadial: int, resolution of radial
                origines: if = '17seg', it follows the form of 17 segments model 
                    else will be set to [0,0] (top left)
        """
        self.numSlices  = numSlices
        self.shapeSlice = shapeSlice
        self.resoRadial = resoRadial
        self.radEpiList = radEpiList
        self.radEndoList = radEndoList
        self.dcm = DicomObj(data = np.full((shapeSlice[0], shapeSlice[1], numSlices),
                                           fill_value=np.nan),
                            offset = offset, spacing = spacing,
                            USliceLocation = np.arange(0,numSlices)*spacing[-1])           
        if maxRadEpi is not None:
            if maxRadEpi > resoRadial:
                raise ValueError('The order should be: maxRadEpi < resoRadial')
            if maxRadEpi is not None:
                if maxRadEpi < maxRadEndo:
                    raise ValueError('The order should be: maxRadEndo < maxRadEpi')
        angs    = np.linspace(0, 2*np.pi, resoRadial+1) + 4/3*np.pi # doesn't matter if add 4/3*pi
        if maxZ is not None:
            zList   = np.linspace(0, maxZ, numSlices)
            radList = (0.5 * (1-np.flip(zList/1)**2))**(1/2)  # oval form
        if self.radEpiList is None:
            self.radEpiList  = maxRadEpi * radList
        if self.radEndoList is None:
            self.radEndoList = maxRadEndo * radList                   
        contours = [ {'non-MI': [np.vstack((self.radEpiList[i]*np.sin(angs) + shapeSlice[0]/2, \
                                            self.radEpiList[i]*np.cos(angs) + shapeSlice[1]/2 )).T, \
                                 np.vstack((self.radEndoList[i]*np.sin(angs) + shapeSlice[0]/2, \
                                            self.radEndoList[i]*np.cos(angs) + shapeSlice[1]/2 )).T ],
                      'MI': [], 'NR': []} \
                    for i in range(numSlices)]
        self.roi = RoiObj(contours=contours)
        self.roi.add_segments(canvasShape = shapeSlice)
        self.roi.add_index_apex_valve(idx_apex=0, idx_valve=numSlices-1) # first slice Z=0, last Z=1
        self.roi.add_endoCenters(mtdEndoCenter_mixEndoEpi = mtdEndoCenter_mixEndoEpi)

        if origines == '17seg':
            origines = funCoord.get_origine_from_angle(self.roi.contours, 
                                                       self.roi.endoCenters,
                                                       angle=np.pi*2/3)
        else:
            origines=np.zeros((2,numSlices))
        self.roi.add_origines_myoOpenings(origines=origines,
                                          myoOpenings=np.full((2,2,numSlices),fill_value=np.nan),
                                          adjust=False)
        self.dcm.resample(samplingFactor, mtdInterp=mtdInterpImg)
        self.roi.resample(samplingFactor, mtdInterp=mtdInterpContour)
        self.roi.add_coordinates(self.dcm)
        
    def get_myo_radials(self, resampled = True):
        """ get radials of myo endo & epi of each slices 
                attention: not the real radial but pixel united one,
                    the radial is computed as the everage distance of myo contour to endo center
            Parameters:
                resampled: if True, compute radial of resampled myo, else, return original 
            return:
                radials: dict {'endo':[], 'epi':[]} contains the radials 
        """
        radials = {'endo':[], 'epi':[]}
        for i in range(self.numSlices):
            if resampled:
                myo = self.roi.contoursResampled[i]['non-MI']
                center = self.roi.endoCentersResampled[:,i]
            else:
                myo = self.roi.contours[i]['non-MI']
                center = self.roi.endoCenters[:,i]
            radials['epi'].append(np.linalg.norm(myo[0]-center,axis=1).mean())
            radials['endo'].append(np.linalg.norm(myo[1]-center,axis=1).mean())
        return radials
        
        
class Model17SegObj():
    def __init__(self, ref=None):
        """ 17 segments model 
            Parameters:
                ref: LV reference
        """
        self.numSegments = 17   
        self.angleOrigine = np.pi*2/3 # counterclockwise from est
        self.type_longAxisSection = ['apex','apical','mid','basal']
        self.type_region = {'LAD': [1,2,7,8,13,14,17],
                            'RCA': [3,4,9,10,15],
                            'LCX': [5,6,11,12,16]}
        if ref is not None:
            self.ref = ref
            self.centers = ref.roi.endoCentersResampled
            self.origines = funCoord.get_origine_from_angle(ref.roi.contoursResampled, 
                                                            ref.roi.endoCentersResampled, 
                                                            angle=self.angleOrigine)
            self.sliceId_in_longAxisSection = self.get_sliceId_in_longAxisSection()
            self.myoRadials = self.ref.get_myo_radials(resampled = True)
        
    @staticmethod    
    def get_segments_info(idxSeg, ratioAnterior=[1,6,6,8]):
        """ get 17 model segments location information
            Parameters:
                idxSeg: segment index, int [1,17]
                ratioAnterior: ratio list [apex, apical, mid, basal]
            Return:
                angle: [2,] ndarray contains the start and the end angle of the segment
                z: [2,] ndarray contains the start and the end z value (long axis) of the segment
        """
        if isinstance(idxSeg, int):
            if idxSeg < 0 or idxSeg > 17:
                raise ValueError('Input idxSeg should be int in [1,17]')
            if idxSeg < 7:
                angle = np.pi*(np.array([-1/3,0]) + (idxSeg-1)/3)
                z = np.array([sum(ratioAnterior[0:3]), sum(ratioAnterior)])/sum(ratioAnterior)
            elif idxSeg >= 7 and idxSeg < 13:
                angle = np.pi*(np.array([-1/3,0]) + (idxSeg-7)/3)
                z = np.array([sum(ratioAnterior[0:2]), sum(ratioAnterior[0:3])])/sum(ratioAnterior)
            elif idxSeg == 17:
                angle = np.array([0, 2*np.pi])
                z = np.array([0, ratioAnterior[0]])/sum(ratioAnterior)
            else : # 13-16
                angle = np.pi*(np.array([-5/12,1/12]) + (idxSeg-13)/2)
                z = np.array([ratioAnterior[0], sum(ratioAnterior[0:2])])/sum(ratioAnterior)
        return angle, z
    
    def get_sliceId_in_longAxisSection(self):
        """ get index of slices located in sections (long Axis section)
            Return:
                sliceId_in_longAxisSection: dict of 4 elements,
                    contains slice indexes in each long axis section
        """
        sliceId_in_longAxisSection = {key: None for key in self.type_longAxisSection}
        segments_represent = {'apex':17,'apical':13,'mid':7,'basal':1}
        z_vals = np.array(self.ref.roi.get_z_vals()) 
        for key, val in segments_represent.items():  
            z_range = self.get_segments_info(val)[1]
            if key == 'basal':
                sliceId_in_longAxisSection[key] = np.where(np.array([z_vals>=z_range[0], 
                                                                     z_vals<=z_range[1]]).all(axis=0))[0]
            else:
                sliceId_in_longAxisSection[key] = np.where(np.array([z_vals>=z_range[0], 
                                                                     z_vals<z_range[1]]).all(axis=0))[0]
        return sliceId_in_longAxisSection
    
    def get_separations_inSlice(self, idxSlice):
        """ get the angle that seperates the segments in the slice
            Return:
                angleSeparation: [n,] ndarray of seperation angles
                lineSeparation: 
        """
        if idxSlice in self.sliceId_in_longAxisSection['apex']:
            angleSeparation = np.array([])
        elif idxSlice in self.sliceId_in_longAxisSection['apical']:
            angleSeparation = (np.array([0,1/2,1,3/2]) + 1/12)*np.pi 
        elif idxSlice in self.sliceId_in_longAxisSection['mid'] or \
            idxSlice in self.sliceId_in_longAxisSection['basal']:
            angleSeparation = np.array([0,1/3,2/3,1,4/3,5/3])*np.pi
        else:
            raise ValueError('Input idxSlice not exist in the reference')
        lineSeparation = []
        for i in angleSeparation:
            sepa_tmp = funCoord.get_euclid_from_polar(i, np.array([self.myoRadials['epi'][idxSlice], 
                                                                   self.myoRadials['endo'][idxSlice]]),
                                                      center=self.centers[:,idxSlice], 
                                                      origineAngle=self.angleOrigine)
            lineSeparation.append(sepa_tmp)
        return angleSeparation, lineSeparation
    
    def get_rotate_angle(self):
        """ get rotation angle (counterclockwise)
            Return:
                angle_diff: the angle between origine of 17 segments model and the reference
        """
        vect_ref_origines = self.centers - self.ref.roi.originesResampled
        angles_ref_origins = np.arctan2(vect_ref_origines[1,:], vect_ref_origines[0,:])
        angles_ref_origins[angles_ref_origins<0] += 2*np.pi
        
        vect_origines = self.centers - self.origines 
        angles_origins = np.arctan2(vect_origines[1,:], vect_origines[0,:])
        angles_origins[angles_origins<0] += 2*np.pi
        
        angle_diff = angles_origins - angles_ref_origins
        return angle_diff
    
    def fit_coordinates(self):
        """ convert coordinates in reference to 17segment coordinates
            Return:
                coordinates_rotated: rotated coordinates
        """
        angle_diff = self.get_rotate_angle()
        coordinates_rotated = np.full(self.ref.roi.coordinates.shape, fill_value=np.nan)
        for i in range(coordinates_rotated.shape[-1]):
            for j in range(coordinates_rotated.shape[-2]):
                coordinates_rotated[:,:,j,i] = funImg.rotate_img(self.ref.roi.coordinates[:,:,j,i], 
                                                                 angle_diff[i], 
                                                                 center=self.centers[:,i])
        # the interpolation of rotation could create some unexpected median value near the origine angle.
        coordinates_rotated = self._check_angle_near_origine(coordinates_rotated)
        return coordinates_rotated       
    
    def _check_angle_near_origine(self, coordinates, margin_aroundOrigine = 5):
        (H, W) = coordinates.shape[:2]
        for i in range(coordinates.shape[-1]):
            for h in range(H):
                for w in range(W):
                    if not np.isnan(coordinates[h,w,1,i]):
                        # dist_tmp = geo.dist_point_to_line(np.array([h,w]), 
                        #                                   np.vstack((self.origines[:,i],
                        #                                              self.centers[:,i])).T)
                        # if dist_tmp <= margin_aroundOrigine:
                        angle_tmp = funCoord.compute_angles_inSlice(np.atleast_2d([h,w]), 
                                                                    self.origines[:,i], 
                                                                    self.centers[:,i])
                        if angle_tmp < margin_aroundOrigine or np.pi*2 - angle_tmp < margin_aroundOrigine:
                                coordinates[h,w,1,i] = angle_tmp
        return coordinates
    
    def fit(self, data=None, segments=None):
        """ fit normalized data, segments & coordinates to 17 segment model
            Parameters:
                data: normalized data
                segments: normalized segmentations instance of roiObj.segments
            Return:
                listReturn: list of rotated data & segments (if not none passed by parameters)
        """
        angle_diff = self.get_rotate_angle()
        listReturn = []
        if data is not None:
            data_rotated = np.full(data.shape, fill_value=np.nan)
            for i in range(data.shape[-1]):
                data_rotated[:,:,i] = funImg.rotate_img(data[:,:,i], angle_diff[i], 
                                                        center=self.centers[:,i])
            listReturn.append(data_rotated)
        if segments is not None:
            segments_rotated = [{} for i in range(len(segments))]
            for i in range(len(segments)):
                for key, val in segments[i].items():
                    segments_rotated[i][key] = funImg.rotate_img(val, angle_diff[i], center=self.centers[:,i])
            listReturn.append(segments_rotated)
        if len(listReturn) < 2:
            return listReturn[0]
        else:
            return listReturn
        

class SegCollectionObj():
    def __init__(self, listSegments=None):
        """ 
            Parameters: 
                listSegments: dict of segmentations with key: case name, vals: segmentations
        """
        self.listSegments = {}
        self.posPolygons = None

    def add_case(self, caseName, val):
        self.listSegments[caseName] = val
        return
        
    def numCases(self):
        return len(self.listSegments.values())
    
    def numSlices(self):
        if len(self.listSegments) > 0:
            return len(list(self.listSegments.values())[0])
        else: 
            return 0
        
    def caseNames(self):
        return list(self.listSegments.keys())       
        
    def labels(self):
        if len(self.listSegments) > 0:
            return list(list(self.listSegments.values())[0][0].keys())
        else:
            return []
    
    def get_cases(self, cases):
        cases_selected = []
        for i in cases:
            cases_selected.append(self.listSegments[i])
        return cases_selected
    
    def fuse_cases(self, cases, infarctType, positions, 
                   idxSlice=None,  fuseAllTypes=1, replaceNan=0):
        """
            fuse the data in myocardium of all cases to ndarray
            or if fuseAllTypes = 1, return a sigle ndarray fuse all infarctType 
            Parameters:
                cases: list of case names
                infarctType: list of infarct types
                positions: list of positions in each slice where the corresponding 
                    will be fused
                idxSlice: list of idx of slices to fuse
                    if None, all slices are fused
                fuseAllTypes: if 1 fuse all types
            Return:
                dict of fused data: fused[l] is an adarray of shape (N*M)
                    N: subject number, M: dimension number
        """
        if idxSlice is None:
            idxSlice = list(range(len(positions)))
        fused = {l:[] for l in infarctType}
        for l in infarctType:
            for i in cases:
                data_tmp = []
                for z in idxSlice:
                    slice_tmp = self.listSegments[i][z][l][positions[z]]
                    if np.isnan(slice_tmp).sum() > 0:
                        slice_tmp[np.isnan(slice_tmp)] = replaceNan
                    #     warnings.warn("case {:s}, slice {:d} has nan value".format(i,z))
                    data_tmp = np.hstack((data_tmp, slice_tmp))
                fused[l].append(data_tmp)   
            fused[l] = np.asarray(fused[l])
        if fuseAllTypes:
            fused = np.hstack(list(fused.values()))
        return fused
        
    def get_mean(self, cases=None):
        """ 
            Parameters: 
                cases: list of case names that will be concerned for mean calculation
        """
        if cases is None:
            cases = self.caseNames()
        labels = self.labels()
        segAvg = []
        for i in range(self.numSlices()):
            seg_tmp = {l: bStats.nanmean(np.array([self.listSegments[j][i][l] for j in cases]), 
                                         axis=0)
                       for l in labels}
            segAvg.append(seg_tmp)        
        return segAvg
    
    def get_variance(self, cases=None):
        """ 
            Parameters: 
                cases: list of case names that will be concerned for mean calculation
        """
        if cases is None:
            cases = self.caseNames()
        labels = self.labels()
        segAvg = []
        for i in range(self.numSlices()):
            seg_tmp = {l: bStats.nanvar(np.array([self.listSegments[j][i][l] for j in cases]), 
                                        axis=0)
                       for l in labels}
            segAvg.append(seg_tmp)        
        return segAvg
    
    def get_p_value(self, cases_a, cases_b, 
                    nan_policy='omit', alternative='two-sided'):
        a = self.get_cases(cases_a)
        b = self.get_cases(cases_b)
        labels = self.labels()
        pValue = []
        for i in range(self.numSlices()):
            p_tmp = {l: bStats.p_value(a=np.array([s[i][l] for s in a]), 
                                       b=np.array([s[i][l] for s in b]), 
                                       axis=0, nan_policy=nan_policy,
                                       alternative=alternative)[1]
                        for l in labels}
            pValue.append(p_tmp)
        return pValue
    
    def get_lesion_area_percentage(self, cases=None, weighted=0):
        lesionArea = {}
        if cases is None:
            cases = self.caseNames()
        for case in cases:
            lesionArea[case] = self.get_lesion_area_of_subject(self.listSegments[case],
                                                               self.labels(),
                                                               weighted=weighted)
        return lesionArea
    
    def get_endo_lesion_area_percentage(self, coordinate, 
                                        cases=None, endoRad = 0.25):
        lesionArea = {}
        if cases is None:
            cases = self.caseNames()
        pos = [coordinate[:,:,0,i] < endoRad for i in range(self.numSlices())]
        for case in cases:
            lesionArea[case] = \
                self.get_endo_lesion_area_of_subject(self.listSegments[case],
                                                     positions = pos, 
                                                     lesionType = self.labels())
        return lesionArea
    
    def get_transmurality(self, coordinate, cases=None, 
                          sliceNeglect=0, 
                          mtd_trans='surface_percent',
                          mtd_scalar='mean'):
        """ 
            Parameters:
                coordinate: the coordinate of reference,
                sliceNeglect: neglect slice numbers close to apex
                mtd_trans: method to compute transmurality in polygon 
                mtd_scalar: rule ('mean'/'max') to conclude the scalar of transmurality of 
                    a subject over transmurality of his polygons
        """
        transm_scalar = {}; transm_polygon = {}
        if cases is None:
            cases = self.caseNames()
        if self.posPolygons is None:
            raise AttributeError("positions of polygons have not been computed " + \
                                 "use get_polygon_positions() to get it.")
        for case in cases:
            transm_scalar[case], transm_polygon[case] = \
                self.get_transmurality_of_subject(self.listSegments[case],
                                                  radCoord = coordinate[:,:,0,:],
                                                  positions = self.posPolygons,
                                                  lesionType = self.labels(), 
                                                  sliceNeglect = sliceNeglect, 
                                                  mtd_trans = mtd_trans,
                                                  mtd_scalar=mtd_scalar)
        transm = {l: np.array([transm_polygon[case][l] for case in cases]).mean(axis=0) \
                  for l in self.labels()}
        return transm_scalar, transm

    def get_polygon_positions(self, coordinate, resAng=120):
        """
            Parameters:
                coordinate: the coordinate of reference,
                resAng: angle resolution for defining polygons
        """
        ang_diff = 2*np.pi/resAng
        self.posPolygons = []
        for i in range(coordinate.shape[-1]): # get positions
            coord = coordinate[:,:,:,i]
            pos_inSlice = []
            for j in range(resAng):
                angle_bounds = np.array([0, ang_diff])+ang_diff*j
                pos_inSlice.append(np.logical_and(coord[:,:,1]<max(angle_bounds),
                                                  coord[:,:,1]>=min(angle_bounds)))
            self.posPolygons.append(pos_inSlice)
        return 
    
    def get_bullsEye_individual(self, infarctTypes, cases=None, mtd='mean'):
        if cases is None:
            cases = self.caseNames()
        if mtd == 'max':
            funScalar = np.nanmax
        elif mtd == 'mean':
            funScalar = np.nanmean
        else:
            raise ValueError('unrecognized mtd')
        if self.posPolygons is None:
            raise AttributeError("positions of polygons have not been computed " + \
                                 "use get_polygon_positions() to get it.")
        resAng = len(self.posPolygons[0])
        bullsEye = {case: {l:[0]*resAng*self.numSlices() for l in infarctTypes} for case in cases}
        for i in range(self.numSlices()):
            for j in range(resAng):
                pos = self.posPolygons[i][j]
                for case in cases:
                    for l in infarctTypes:
                        if not np.all(np.isnan(self.listSegments[case][i][l][pos])):
                            bullsEye[case][l][j+i*resAng] = funScalar(self.listSegments[case][i][l][pos])
        return bullsEye
    
    def get_bullsEye_stats(self, infarctType, mean, transmurality,
                           variance, mtd_trans = 'meanOfTrans'):
        """
            Parameters:
                mtd_trans: "meanOfTrans" -- mean of polygonwised transmuralities over subjects
                            "transOfMean" -- polygonwised transmuralities of pixelwised mean over subjects
        """
        resAng = len(self.posPolygons[0])
        locVals = resAng*self.numSlices()
        statsBullsEye = {'location': [0]*locVals,
                         'transmurality': [0]*locVals,
                         'variability': [0]*locVals} 
        for i in range(self.numSlices()):
            for j in range(resAng):
                idx_tmp = j+i*resAng
                pos = self.posPolygons[i][j]
                if not np.all(np.isnan(mean[i][infarctType][pos])):
                    statsBullsEye['location'][idx_tmp] = np.nanmean(mean[i][infarctType][pos])
                if not np.all(np.isnan(variance[i][infarctType][pos])):
                    statsBullsEye['variability'][idx_tmp] = np.nanmean(np.sqrt(variance[i][infarctType][pos])) # std
        # deal with transmurality            
        if mtd_trans == 'meanOfTrans':
            statsBullsEye['transmurality'] = transmurality[infarctType]
        elif mtd_trans == "transOfMean":
            for i in range(self.numSlices()):
                for j in range(resAng):
                    idx_tmp = j+i*resAng
                    pos = self.posPolygons[i][j]
                    if np.sum(mean[i][infarctType][pos]>0)>0:
                            statsBullsEye['transmurality'][idx_tmp] = \
                                np.nanmean(mean[i][infarctType][pos][mean[i][infarctType][pos]>0])
        else:
            raise TypeError("mtd_trans: {:s} is not recognised".format(mtd_trans))
        return statsBullsEye
    
    def get_bullsEye_pValue(self, infarctType, pValue_pixelwise):
        resAng = len(self.posPolygons[0])
        p_value = [1]*resAng*self.numSlices()
        for i in range(self.numSlices()):
            for j in range(resAng):
                idx_tmp = j+i*resAng
                pos = self.posPolygons[i][j]
                if not np.all(np.isnan(pValue_pixelwise[i][infarctType][pos])):
                    p_value[idx_tmp] = np.nanmin(pValue_pixelwise[i][infarctType][pos])
        return p_value
                
    
    @staticmethod
    def get_endo_lesion_area_of_subject(segment, positions, lesionType):
        lesionPercent = {}
        for l in lesionType:
            lesionArea = []; myoArea = []
            for i in range(len(segment)):
                lesionArea.append(np.sum(segment[i][l][positions[i]]>0))
                myoArea.append(np.sum(positions[i]))
            lesionPercent[l] = np.sum(lesionArea)/np.sum(myoArea)
        return lesionPercent
    
    @staticmethod
    def get_lesion_area_of_subject(segment, lesionType, weighted=0):
        lesionPercent = {}
        for l in lesionType:
            lesionArea = []; myoArea = []
            if weighted:
                for i in range(len(segment)):
                    lesionArea.append(np.sum(segment[i][l][segment[i][l]>0]))
                    myoArea.append(np.sum(~np.isnan(segment[i][l])))
            else:                       
                for i in range(len(segment)):
                    lesionArea.append(np.sum(segment[i][l]>0))
                    myoArea.append(np.sum(~np.isnan(segment[i][l])))
            lesionPercent[l] = np.sum(lesionArea)/np.sum(myoArea)
        return lesionPercent
    
    @staticmethod 
    def get_transmurality_of_subject(segment, radCoord, positions, lesionType,
                                     sliceNeglect, mtd_trans, mtd_scalar):
        """
            Parameters: 
                'surface_percent': Nb of infarcted coordinates / total amount of coordinates in polygon
                'radial_percent': Nb of infarcted unique radial coordinates / total amount of unique radial cooridinates
        """
        transm_scalar = {}; transm = {}
        for l in lesionType:
            transm[l] = [] # collect transmurality in polygons of the same subject
            if mtd_trans == 'radial_percent':
                for i in range(sliceNeglect, len(segment)):
                    for pos in positions[i]:
                        rad_infacted = np.unique(radCoord[:,:,i][pos][segment[i][l][pos]>0])
                        transm[l].append(len(rad_infacted)/len(np.unique(radCoord[:,:,i][pos])))
            elif mtd_trans == 'surface_percent':
                for i in range(sliceNeglect, len(segment)):
                    for pos in positions[i]:
                        surf_infacted = np.sum(segment[i][l][pos]>0)
                        transm[l].append(surf_infacted/np.sum(pos))
            else:
                raise NotImplementedError("the mtd_trans is not implemented")
            transm_tmp = np.array(transm[l])
            if mtd_scalar == "max":
                transm_scalar[l] = transm_tmp[transm_tmp>0].max() if np.sum(transm_tmp>0)>0 else 0
            elif mtd_scalar == "mean":
                transm_scalar[l] = transm_tmp[transm_tmp>0].mean() if np.sum(transm_tmp>0)>0 else 0
            else:
                raise TypeError("method {:s} is not defined".format(mtd_scalar))
        return transm_scalar, transm
    


        
    

    
             
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
            
            
