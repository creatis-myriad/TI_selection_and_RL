#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:19:30 2020

@author: zheng
"""

# import numpy as np
import os
import glob

def find_filenames(folder, strPattern, loc='anywhere'):
    """ find filenames (or subfolder names) in folder which have their names match some string pattern
        Paramters: 
            folder: string 
            strPattern: the string pattern that is checked 
            loc: string, ocation of the pattern in the file name, 
                can be 'end', 'start', 'anywhere' (default)
        Returns:
            list of filenames satisfing the condition
    """
    if loc == 'end':
        file = [fileName for fileName in os.listdir(folder) \
                if fileName.endswith(strPattern)]
    elif loc == 'begin':
        file = [fileName for fileName in os.listdir(folder) \
                if fileName.startswith(strPattern)]
    elif loc == 'anywhere':
        file = [fileName for fileName in os.listdir(folder) \
                if fileName.find(strPattern)>-1]
    else:
        raise TypeError('"loc" should be "end", "start" or "anywhere".')
    return file

def GetPaths(input_dir, name_files=["dcm.pkl", "roi.pkl"], match_patern=None) :
    normpath = os.path.normpath("/".join([input_dir, '**', '*']))
    paths = {}
    for elem in name_files :
        paths[elem] = []

    for file in glob.iglob(normpath, recursive=True):
        if os.path.isfile(file) and True in [name in file for name in name_files]:
            if match_patern != None and match_patern not in file : continue
            for elem in name_files :    
                if elem in file :
                    paths[elem].append(file)

    return paths