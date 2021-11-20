#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# src » main.py
# ==============================================
# @Author    : Muhammad Wafi <mwafi@mwprolabs.com>
# @Support   : [www.mwprolabs.com]
# @Created   : 29-05-2019
# @Modified  : 29-05-2019 23:53:37 pm
# ----------------------------------------------
# @Copyright (c) 2021 MWprolabs www.mwprolabs.com
# 
###

import cv2
import glob
import random
import numpy as np
from sklearn.svm import SVC
import numpy as np
import pandas as pd
# local
from local_binary_pattern import LocalBinaryPattern
from facial_landmark import FacialLandMark
from settings import conf

class FaceRecognition():
    
    def __init__(self, emotions, c_value, gamma):
        self.emotion_lists = emotions
        # get LBP class
        self.lbp = LocalBinaryPattern(points=24, radius=8)
        # get FL class
        self.facialLandmark = FacialLandMark()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # svm
        self.svm_clf = SVC(C=c_value, gamma=gamma, kernel='linear', probability=True, tol=1e-3)
        self.c_value = c_value
        self.gamma = gamma
        # temp
        self.data = {}

    # Get all image dataset
    def get_dataset(self, emotion):
        files = glob.glob("datasets\\%s\\*" %emotion)
        random.shuffle(files)
        # split train and test data
        training = files[:int(len(files)*conf.TRAINING)]
        prediction = files[-int(len(files)*conf.TESTING):]
        return training, prediction

    # Get feature extraction
    def extract_feature(self):
        training_feature = []
        training_label = []
        prediction_feature = []
        prediction_label = []

        for emotion in self.emotion_lists:
            print(f"...getting on {emotion}")
            training, prediction = self.get_dataset(emotion)

            # TRAINING
            for item in training:
                image = cv2.imread(item)
                #convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe_image = self.clahe.apply(gray)
                # extract feature using lbp and facial landmark
                fl_result = self.facialLandmark.get_landmarks(clahe_image)
                lbp_result = self.lbp.get_lbp(clahe_image)
                # -- training
                training_feature.append(np.append(fl_result, lbp_result))
                training_label.append(self.emotion_lists.index(emotion))
            
            # PREDICTION
            for item in prediction:
                image = cv2.imread(item)
                #convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe_image = self.clahe.apply(gray)
                # extract feature using lbp and facial landmark
                fl_result = self.facialLandmark.get_landmarks(clahe_image)
                lbp_result = self.lbp.get_lbp(clahe_image)
                # -- testing
                prediction_feature.append(np.append(fl_result, lbp_result))
                prediction_label.append(self.emotion_lists.index(emotion))
            
            # Convert data to pandas dataframe
            # for readable data
            # training_dt = pd.DataFrame(list(zip(training_feature, training_label)), 
            # columns=['Features','Labels'])
            # prediction_dt = pd.DataFrame(list(zip(prediction_feature, prediction_label)), 
            # columns=['Features','Labels'])

        # # Save extracted feature to .xlsx format
        # self.saveResult(training_dt, conf.RESULT_TRAINING)
        # self.saveResult(prediction_dt, conf.RESULT_TESTING)

        return training_feature, training_label, prediction_feature, prediction_label
    
    # Classification using SVM
    def run(self, test_mode):
        accuracy = []
        # Run process 10 times to get mean accuracy
        for i in range(0,10):
            print(f"Collecting datasets ({i}-10):")
            training_data, training_labels, prediction_data, prediction_labels = self.extract_feature()
            # Set training data to numpy arr
            train_features = np.array(training_data)
            # pprint(train_features)
            # train SVM
            self.svm_clf.fit(train_features, training_labels)
            # get accuracy by score() func
            test_features = np.array(prediction_data)
            result = self.svm_clf.score(test_features, prediction_labels)
            print(f"Accuracy: {result}")
            print("-----------------------------\n")
            # Store accuracy in a list
            accuracy.append(result)
            # get mean accuracy
            mean_accuracy = np.mean(accuracy)
            self.testing(test_mode=test_mode, score=result)
            
        return result
    
    # Saving formatted data to excel
    def saveResult(self, data, file_path):
        try:
            writer = pd.ExcelWriter(file_path)
            data.to_excel(writer)
            writer.save()
            print("File saved at {}".format(file_path))
            return True
        except:
            print("Error while saving result file!")
            return False

    # Test result
    def testing(self, test_mode, score):
        write_format = "C:{0} Gamma:{1} Accuracy:{2}".format(self.c_value, self.gamma, score)
        # c_value
        if test_mode == "c_value":
            testing_file = "testing/c_value.txt"
        # gamma
        elif test_mode == "gamma":
            testing_file = "testing/gamma.txt"
        # global
        elif test_mode == "global":
            testing_file = "testing/global.txt"
        # save result for testing purpose
        try:
            with open(testing_file, "a") as outfile:
                outfile.write(write_format + "\n")
        except IOError:
            print("Cannot write to a file!")


if __name__ == "__main__":
    # Emotion lists by datasets
    emotion_lists = [
        "Anger", 
        "Disgust", 
        "Fear", 
        "Happy", 
        "Neutral", 
        "Sadness",
        "Surprised"
    ]

    # Set kernel type and tol (by default using linear, and tol=1e-3)
    c_list      = [1e1, 1e2, 1e3, 1e4, 1e5, 5e3, 5e4]
    gamma_list  = [0.0001, 0.001, 0.01, 0.0005, 0.005, 0.1]

    # for i in gamma_list:
    fer = FaceRecognition(emotion_lists, c_value=5e4, gamma=0.1)
    result = fer.run(test_mode="global")
    