#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:39:30 2020

@author: zheng
"""

# import time
# import threading
# import matplotlib.pyplot as plt

def get_numSlice_apex_valve(tp = 'float'):
    """ Get user input of the slice number of apex and valve
        Parameters:
            tp: type of index 'float' or 'int'
        Returns:
            idx_apex: slice number of apex
            idx_valve: slice number of valve
    """
    idx_apex  = input_with_type('Enter the slice number of APEX' + \
                                '\n(can be negative and float): ', 
                                tp = tp)
    idx_valve = input_with_type('Enter the slice number of VALVE' +
                                '\n(can be negative and float): ',
                                tp = tp)  
    return idx_apex, idx_valve

def input_with_type(prompt , tp='float'):
    while True:
        if tp == 'float':
            try:
                return float(input(prompt))
            except ValueError:
                print('That is not a valid number.')
        elif tp == 'int':
            try:
                return int(input(prompt))
            except ValueError:
                print('That is not a valid integer.')    
        else:
            raise TypeError('tp should be either float or int')
            return
    

# class PromptHack():
#     """ Super Hacky Way of Getting input() to work in Spyder with Matplotlib open
#         No efforts made towards thread saftey!
#         Edited from: 
#         https://stackoverflow.com/questions/34938593/matplotlib-freezes-when-input-used-in-spyder?rq=1
#     """
#     def __init__(self):
#         self.prompt = False
#         self.promptText = ""
#         self.done = False
#         self.waiting = False
#         self.response = ""
#         self.regular_input = input
        
#     def threadfunc(self):    
#         while not self.done:   
#             if self.prompt:   
#                 self.prompt = False
#                 self.response = self.regular_input(self.promptText)
#                 self.waiting = True
#             time.sleep(0.1)
            
#     def input(self, text):
#         self.promptText = text
#         self.prompt = True
#         while not self.waiting:
#             plt.pause(0.01)
#         self.waiting = False
#         return self.response
        
#     def start(self):
#         thread = threading.Thread(target = self.threadfunc)
#         thread.start()
    
#     def finish(self):
#         self.done = True    

#     pass