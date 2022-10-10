from operator import index
import os
import sys; sys.path.insert(0, os.path.abspath("../"))

import collections
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp
thispath = Path(__file__).resolve()

from sklearn import preprocessing
import skimage
from scipy.stats import skew, kurtosis
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from statistics import stdev
from sklearn import linear_model

fos_names = ['mean', 'std','skew', 'kur', 'ent']

color_params = {'spaces': ['rgb', 'lab', 'ycrbcb', 'hsv'],

}

class FeaturesExtraction():
    def __init__(
        self, levels: List[str] = ['global', 'local'],
        color_params: dict = color_params,
        n_jobs: int = -1
    ):
        super(FeaturesExtraction, self).__init__()

        self.levels = levels
        self.color_params = color_params

        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        self.n_jobs = n_jobs

        self.get_feature_names()

    def get_feature_names(self):

        self.features_names = []

        if self.color_params:
            for level in self.levels:
                for space in self.color_params['spaces']:
                    for fos in fos_names:
                        for i in range(1,4):
                            self.features_names.extend([f'{level}_{space}_{fos}_{i}'])

    def extract_features(self, image: np.ndarray, mask = np.NaN):

        features = []

        if self.color_params:
            features.extend(self.get_color_features(image, mask))
  
        # other features
        # features.extend(self.get_X_features(image), mask)
        # candidates_features = np.concatenate(
        #         [features_others, features], axis=1)

        return features

    def get_color_features(self, image, mask):

        color_feat = []
        for level in self.levels:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            color_feat.extend(self.get_statistics(img_rgb))
            color_feat.extend(self.get_statistics(img_lab))
            color_feat.extend(self.get_statistics(img_ycrcb))
            color_feat.extend(self.get_statistics(img_hsv))

            if level == 'local':
                # get_local_statistics
                pass

        return color_feat            

    def get_statistics(self, img):
        mean1 = np.mean(img[:,:,0])
        mean2 = np.mean(img[:,:,1])
        mean3 = np.mean(img[:,:,2])
        std1 = np.std(img[:,:,0])
        std2 = np.std(img[:,:,1])
        std3 = np.std(img[:,:,2])
        # Skewness
        val1 = skew((img[:,:,0]).reshape(-1))
        val2 = skew((img[:,:,1]).reshape(-1))
        val3 = skew((img[:,:,2]).reshape(-1))
        # Kurtosis
        kval1 = kurtosis((img[:,:,0]).reshape(-1))
        kval2 = kurtosis((img[:,:,1]).reshape(-1))
        kval3 = kurtosis((img[:,:,2]).reshape(-1))
        # Entropy
        entropy1 = skimage.measure.shannon_entropy(img[:,:,0])
        entropy2 = skimage.measure.shannon_entropy(img[:,:,1])
        entropy3 = skimage.measure.shannon_entropy(img[:,:,2])

        return [mean1,mean2,mean3, std1,std2,std3,val1,val2,val3,kval1,kval2,kval3, entropy1, entropy2, entropy3]
