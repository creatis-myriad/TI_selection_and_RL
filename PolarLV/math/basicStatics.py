#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:05:00 2021

Basic statistics
Mean, variance, Hypothesis test

@author: zheng
"""

import numpy as np
import warnings
import scipy.stats as stats

def nanmean(data, axis=None, dtype=None, out=None, keepdims=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg = np.nanmean(data, axis=axis, 
                         dtype=dtype, out=out, keepdims=keepdims)
    return avg

def nanvar(data, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        var = np.nanvar(data, axis=axis, 
                        dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
    return var

def p_value(a, b, axis=0, nan_policy='omit', alternative='two-sided'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        t, pValue = stats.ttest_rel(a, b, axis = axis, 
                                    nan_policy = nan_policy,
                                    alternative = alternative)
        if hasattr(pValue, 'data'):
            pValue= pValue.data
    return t, pValue