import enum
from operator import index
import os
import sys;

sys.path.insert(0, os.path.abspath("../"))

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
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

fos_names = ['mean', 'std', 'skew', 'kur', 'ent']
glcm_stats = ['contrast','dissimilarity','homogeneity','ASM','energy','correlation']

color_params = {'spaces': ['rgb', 'lab', 'ycrbcb', 'hsv']}

lbp_params = {'radius': [1, 3],'points': 8 }

glcm_params = {'angles' : [0, np.pi/4, np.pi/2, 3*np.pi/4],
                'distance' : [1, 2]}


class FeaturesExtraction():
    def __init__(
            self, levels: List[str] = ['global', 'local'],
            color_params: dict = color_params,
            lbp_params: dict = lbp_params,
            glcm_params: dict = glcm_params,
            n_jobs: int = -1
    ):
        super(FeaturesExtraction, self).__init__()

        self.levels = levels
        self.color_params = color_params
        self.lbp_params = lbp_params
        self.glcm_params = glcm_params

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
                        for i in range(1, 4):
                            self.features_names.extend([f'{level}_{space}_{fos}_{i}'])

        if self.lbp_params:
            for level in self.levels:
                if level != 'local':
                    for rad in self.lbp_params['radius']:
                        for lbp_idx in range(10):
                            self.features_names.extend([f'{level}_rad_{rad}_lbp{lbp_idx}'])
        
        if self.glcm_params:
            for level in self.levels:
                if level != 'local':
                    for feat in glcm_stats:
                        for distance in self.glcm_params['distance']:
                            for angle_idx in range(len(self.glcm_params['angles'])):                        
                                self.features_names.extend([f'{level}_dist{distance}_ang{angle_idx}_{feat}'])
                
        # if self.glcm_params:



    def extract_features(self, image: np.ndarray, mask=None):
        """
        Extract features from an image. Features can be extracted from
        global (whole image) or local (segmented lesion).
        Features: COLOR

        Args:
            image (np.ndarray): Original or preprocessed image (uint8 BGR)
            mask (np.ndarray, optional): If local features is selected,
            the mask (uint8, same dimensions as image) will be used to slice image array.
            Defaults to None.

        Returns:
            List(float32): List of all extracted features
        """
        features = []
    
        if self.color_params:
            features.extend(self.get_color_features(image, mask))

        self.img_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.lbp_params:
            features.extend(self.get_lbp_features(self.img_gs, mask))

        if self.glcm_params:
            features.extend(self.get_glcm_features(self.img_gs))
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

        return [mean1, mean2, mean3, std1, std2, std3, val1, val2, val3, kval1, kval2, kval3, 
                entropy1, entropy2, entropy3]

    def get_lbp_features(self, image: np.ndarray, mask: np.ndarray):
        """
        Obtain LBP features from an image at global or local level.

        Args:
            image (np.ndarray): Original or preprocessed image (uint8 BGR)
            mask (np.ndarray): If local features is selected,
            the mask (uint8, same dimensions as image) will be used to slice image array.

        Returns:
            List(float32): List of all LBP features
        """
        lbp_feat = []

        for level in self.levels:
            if level == 'global':
                # mask_g = np.ones(image.shape[:2])
                lbp_feat.extend(self.get_lbp_calculation(image))
            else:
                continue
                # lbp_feat.extend(self.get_lbp_calculation(image, mask))

        return lbp_feat

    def get_lbp_calculation(self, img: np.ndarray):
        """
        Obtain LBP features for three channel (RGB).

        Args:
            img (np.ndarray): _description_
            mask (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        
        lbp_feature_vector = []

        for rad in self.lbp_params['radius']:
            n_points = self.lbp_params['points'] * rad            
            lbp = local_binary_pattern(img, n_points, rad, method="uniform")
            hist = np.histogram(lbp.ravel())[0].astype(np.float32)
            hist = hist/(hist.sum() + np.finfo(np.float32).eps)
            lbp_feature_vector.extend(np.ndarray.tolist(hist))

        # # lbp from saturation channel (HSV)
        # lbp = local_binary_pattern(self.img_hsv[:,:,1], n_points, self.lbp_params['radius'], method="uniform")
        # hist = np.histogram(lbp.ravel())[0].astype(np.float32)
        # hist = hist/(hist.sum() + np.finfo(np.float32).eps)
        # lbp_feature_vector.extend(np.ndarray.tolist(hist))

        # for ch in range(3):
        #     lbp = local_binary_pattern(img[:,:,ch], n_points, self.lbp_params['radius'], method="uniform")
        #     hist = np.histogram(lbp.ravel())[0].astype(np.float32)
        #     hist = hist/(hist.sum() + np.finfo(np.float32).eps)
        #     lbp_feature_vector.extend(np.ndarray.tolist(hist))

        return lbp_feature_vector

    def get_glcm_features(self, img: np.ndarray):

        glcm_feat = []
        glcm_decomp = graycomatrix(img, self.glcm_params['distance'], 
                                    self.glcm_params['angles'], normed=True)

        for feat_name in glcm_stats:
            glcm_feat.extend(graycoprops(glcm_decomp, feat_name).ravel())
        
        return glcm_feat
        


