# Positive_transform.py
"""
Converted from Positive_transform.m
Provides transforms that enforce positivity.
"""
import numpy as np

def Positive_transform(x):
    # softplus as a smooth positive transform (log(1+exp(x)))
    return np.log1p(np.exp(x))
