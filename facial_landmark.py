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

import dlib
import math
import numpy as np
from settings import conf

class FacialLandMark():

    def __init__(self):
        # predictor set
        self.predictor = dlib.shape_predictor(conf.PREDICTOR)
        self.detector = dlib.get_frontal_face_detector()
        self.data = {}

    def get_landmarks(self, image):
        detections = self.detector(image, 1)
        #For all detected face instances individually
        for k,d in enumerate(detections):
            #Draw Facial Landmarks with the predictor class
            shape = self.predictor(image, d)
            xlist = []
            ylist = []
            # X and Y coordinates in two lists
            for i in range(1,68):
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))
            xmean = np.mean(xlist)
            ymean = np.mean(ylist)
            xcentral = [(x-xmean) for x in xlist]
            ycentral = [(y-ymean) for y in ylist]
            landmarks_vectorised = []

            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorised.append(w)
                landmarks_vectorised.append(z)
                meannp = np.asarray((ymean,xmean))
                coornp = np.asarray((z,w))
                dist = np.linalg.norm(coornp-meannp)
                landmarks_vectorised.append(dist)
                landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
            self.data['landmarks_vectorised'] = landmarks_vectorised

        if len(detections) < 1:
            self.data['landmarks_vestorised'] = "error"

        return landmarks_vectorised
