#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# src » main.py
# ==============================================
# @Author    : Muhammad Wafi <mwafi@mwprolabs.com>
# @Support   : [www.mwprolabs.com]
# @Created   : 28-05-2019
# @Modified  : 28-05-2019 23:53:37 pm
# ----------------------------------------------
# @Copyright (c) 2021 MWprolabs www.mwprolabs.com
# 
###

import numpy as np
from skimage.feature import local_binary_pattern

class LocalBinaryPattern():
    
    def __init__(self, points, radius, eps=1e-7):
        # init vars
        self.radius = radius
        self.points = points
        self.eps = eps
        self.data = {}

    def get_lbp(self, image):
        # get lbp with uniform method
        getLbp = local_binary_pattern(image, self.points, self.radius, method="uniform")
        (histogram, _) = np.histogram(
            getLbp.ravel(), 
            bins=np.arange(0, self.points + 3), 
            range=(0, self.points + 2)
        )
        # normalization
        histogram = histogram.astype("float")
        histogram /= (histogram.sum() + self.eps)

        self.data['lbp_feature'] = histogram

        return histogram
