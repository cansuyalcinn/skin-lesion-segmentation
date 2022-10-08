from operator import index
import os
import sys; sys.path.insert(0, os.path.abspath("../"))

import collections
import logging
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp
thispath = Path(__file__).resolve()

epsillon = np.finfo(float).eps

logging.basicConfig(level=logging.INFO)

# Default parameters

HAIR_REMOVAL_PARAMS = {'kernel_size': (20,20),
                        'gauss_kernel_size': (3,3),
                        'thresh_low': 10,
                        'thresh_high': 255}

class SkinLesionPreprocessing:
    def __init__(self,
                remove_fov: bool = True,
                resize: bool = False,
                hair_removal_params: dict = HAIR_REMOVAL_PARAMS,

    ):
        """ Preprocess the images from Skin Lesion Dataset
        Removes FOV (vignette), reisize the images to a defined length,
        and removes dark hairs from image.

        Args:
            remove_fov (bool, optional): Whether to remove the vignette.
            Defaults to True.
            resize (bool, optional): Whether to resize the image. 
            Defaults to False.
            hair_removal_params (dict, optional): hair removal algorithm parameters. 
            Defaults to HAIR_REMOVAL_PARAMS.
        """
        super(SkinLesionPreprocessing, self).__init__()

        self.remove_fov = remove_fov
        self.resize = resize
        self.hair_removal_params = hair_removal_params

        if resize:
            self.rs_height = 200
            self.rs_width = 200
        
    def preprocess(self, image: np.ndarray):
        
        if self.remove_fov:
            cropped_image = self.crop_image(image)

        if self.resize:
            resized_image = self.resize_image(cropped_image)
        
        

        

        
    def remove_hair(self, image: np.ndarray):
        """
        Removes dark hairs from image with blackhat technique and inpainting.

        Args:
            image (np.ndarray): 3 channel BGR uint8 original image

        Returns:
            np.ndarray: 3 channel BGR uint8 image without hairs
        """
        # get red channel
        red = image[:,:,2]

        # Gaussian filter
        gaussian= cv2.GaussianBlur(red, self.hair_removal_params['gauss_kernel_size'], 
                                    cv2.BORDER_DEFAULT)

        # Black hat filter
        kernel = cv2.getStructuringElement(1, self.hair_removal_params['kernel_size'])
        blackhat = cv2.morphologyEx(gaussian, cv2.MORPH_BLACKHAT, kernel)

        #Binary thresholding (MASK)
        ret,mask = cv2.threshold(blackhat, self.hair_removal_params['thresh_low'], 
                                    self.hair_removal_params['thresh_high'], cv2.THRESH_BINARY)

        # apply opening : erosion+dilation (remove the dots that are captured and then extend the hair parts)
        kernel_opening = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening)
        kernel_dilation = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(opening, kernel_dilation, iterations=1)

        #Replace pixels of the mask
        dst = cv2.inpaint(image,dilated_mask,6,cv2.INPAINT_TELEA)

        return dst

    def resize_image(self, image: np.ndarray):
        pass

    def crop_image(self, image: np.ndarray):
        pass

