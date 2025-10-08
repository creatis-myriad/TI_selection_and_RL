#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:00:16 2021

@author: zheng
"""

def remove_key(d, key):
    """
        remove a key and the value from a dict
    """
    r = dict(d)
    del r[key]
    return r