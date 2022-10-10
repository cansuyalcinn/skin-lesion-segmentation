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
        """
        Extract features from an image. Features can be extracted from
        global (whole image) or local (segmented lesion).
        Features: COLOR

        Args:
            image (np.ndarray): Original or preprocessed image (uint8 BGR)
            mask (np.ndarray, optional): If local features is selected,
            the mask (uint8, same dimensions as image) will be used to slice image array.
            Defaults to np.NaN.

        Returns:
            List(float32): List of all extracted features
        """
        features = []

        if self.color_params:
            features.extend(self.get_color_features(image, mask))

        # other features
        # features.extend(self.get_X_features(image), mask)
        # candidates_features = np.concatenate(
        #         [features_others, features], axis=1)

        return features

    def get_color_features(self, image: np.ndarray, mask: np.ndarray):
        """
        Obtain color features from an image at global or local level.

        Args:
            image (np.ndarray): Original or preprocessed image (uint8 BGR)
            mask (np.ndarray): If local features is selected,
            the mask (uint8, same dimensions as image) will be used to slice image array.
            Defaults to np.NaN.

        Returns:
            List(float32): List of all color features
        """
        color_feat = []

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for level in self.levels:
            if level == 'global':
                mask_g = np.ones(image.shape[:2])
                color_feat.extend(self.get_statistics(img_rgb, mask_g))
                color_feat.extend(self.get_statistics(img_lab, mask_g))
                color_feat.extend(self.get_statistics(img_ycrcb, mask_g))
                color_feat.extend(self.get_statistics(img_hsv, mask_g))
            else:                
                color_feat.extend(self.get_statistics(img_rgb, mask))
                color_feat.extend(self.get_statistics(img_lab, mask))
                color_feat.extend(self.get_statistics(img_ycrcb, mask))
                color_feat.extend(self.get_statistics(img_hsv, mask)) 

        return color_feat            

    def get_statistics(self, img: np.ndarray, mask: np.ndarray):
        """
        Obtain local first order statistics from an image by slicing it with 
        a boolean mask

        Args:
            img (np.ndarray): _description_
            mask (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        img = img.astype(np.float32)
        mask = mask.astype(bool)

        mean1 = np.mean(img[mask, 0])
        mean2 = np.mean(img[mask, 1])
        mean3 = np.mean(img[mask, 2])
        
        std1 = np.std(img[mask, 0])
        std2 = np.std(img[mask, 1])
        std3 = np.std(img[mask, 2])
        # Skewness
        val1 = skew((img[mask, 0]).reshape(-1))
        val2 = skew((img[mask, 1]).reshape(-1))
        val3 = skew((img[mask, 2]).reshape(-1))
        # Kurtosis
        kval1 = kurtosis((img[mask, 0]).reshape(-1))
        kval2 = kurtosis((img[mask, 1]).reshape(-1))
        kval3 = kurtosis((img[mask, 2]).reshape(-1))
        # Entropy
        entropy1 = skimage.measure.shannon_entropy(img[mask, 0])
        entropy2 = skimage.measure.shannon_entropy(img[mask, 1])
        entropy3 = skimage.measure.shannon_entropy(img[mask, 2])

        return [mean1,mean2,mean3, std1,std2,std3,val1,val2,val3,kval1,kval2,kval3, entropy1, entropy2, entropy3]
