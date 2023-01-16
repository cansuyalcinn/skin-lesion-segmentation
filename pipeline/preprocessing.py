from operator import index
import os
import sys;

sys.path.insert(0, os.path.abspath("../"))

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

HAIR_REMOVAL_PARAMS = {'kernel_size': (20, 20),
                       'gauss_kernel_size': (3, 3),
                       'thresh_low': 10,
                       'thresh_high': 255}


class SkinLesionPreprocessing:
    def __init__(self,
                 remove_fov: bool = True,
                 resize: bool = False,
                 hair_removal_params: dict = HAIR_REMOVAL_PARAMS,
                 remove_hair: bool = True,

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
            remove_hair (bool, optional): Whether to remove hair
        """
        self.remove_fov = remove_fov
        self.resize = resize
        self.hair_removal_params = hair_removal_params
        self.remove_hair = remove_hair

    def preprocess(self, image: np.ndarray, resize_shape=(225, 300)):

        # if 'size' not in md_df.columns:
        #     md_df['size']

        if self.remove_fov:
            image_preproc = self.crop_image(image)

            # case where the image cropping fails, choose not to crop
            if not image_preproc.any():
                image_preproc = image

        if self.resize:
            image_preproc = self.resize_image(image_preproc, resize_shape)
        
        if self.remove_hair:
            image_preproc = self.hair_removal(image_preproc)

        return image_preproc

    def hair_removal(self, image: np.ndarray):

        """
        Removes dark hairs from image with blackhat technique and inpainting.

        Args:
            image (np.ndarray): 3 channel BGR uint8 original image

        Returns:
            np.ndarray: 3 channel BGR uint8 image without hairs
        """
        # get red channel
        red = image[:, :, 2]

        # Gaussian filter
        gaussian = cv2.GaussianBlur(red, self.hair_removal_params['gauss_kernel_size'],
                                    cv2.BORDER_DEFAULT)

        # Black hat filter
        kernel = cv2.getStructuringElement(1, self.hair_removal_params['kernel_size'])
        blackhat = cv2.morphologyEx(gaussian, cv2.MORPH_BLACKHAT, kernel)

        # Binary thresholding (MASK)
        ret, mask = cv2.threshold(blackhat, self.hair_removal_params['thresh_low'],
                                  self.hair_removal_params['thresh_high'], cv2.THRESH_BINARY)

        # Apply opening : erosion+dilation (remove the dots that are captured and then extend the hair parts)
        kernel_opening = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening)
        kernel_dilation = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(opening, kernel_dilation, iterations=1)

        # Replace pixels of the mask
        dst = cv2.inpaint(image, dilated_mask, 6, cv2.INPAINT_TELEA)

        return dst

    def resize_image(self, image: np.ndarray, resize_shape: tuple):

        # Add condition about aspect ratio

        return cv2.resize(image, resize_shape, interpolation=cv2.INTER_CUBIC)

    # def resize_with_pad(self, image: np.array, 
    #                 new_shape: Tuple[int, int], 
    #                 padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    #     """Maintains aspect ratio and resizes with padding.
    #     Params:
    #         image: Image to be resized.
    #         new_shape: Expected (width, height) of new image.
    #         padding_color: Tuple in BGR of padding color
    #     Returns:
    #         image: Resized image with padding
    #     """
    #     original_shape = (image.shape[1], image.shape[0])
    #     ratio = float(max(new_shape))/max(original_shape)
    #     new_size = tuple([int(x*ratio) for x in original_shape])
    #     image = cv2.resize(image, new_size)
    #     delta_w = new_shape[0] - new_size[0]
    #     delta_h = new_shape[1] - new_size[1]
    #     top, bottom = delta_h//2, delta_h-(delta_h//2)
    #     left, right = delta_w//2, delta_w-(delta_w//2)
    #     image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    #     return image

    def crop_image(self, image: np.ndarray, threshold=100):
        """
        Crop the image to get the region of interest. Remove the vignette frame.
        Analyze the value of the pixels in the diagonal of the image, from 0,0 to h,w and
        take the points where this value crosses the threshold by the first time and for last.
        
        Args:
        - img (numpy ndarray): Image to crop.
        - threshold (int): Value to split the diagonal into image and frame.

        Returns:
            np.ndarray: Cropped image
            tuple: Shape of cropped image
        """
        # Get the image dimensions
        h, w = image.shape[:2]

        # Get the coordinates of the pixels in the diagonal
        if h != 1024:
            y_coords = ([i for i in range(0, h, 3)], [i for i in range(h - 3, -1, -3)])
        y_coords = ([i for i in range(0, h, 4)], [i for i in range(h - 4, -1, -4)])
        x_coords = ([i for i in range(0, w, 4)], [i for i in range(0, w, 4)])

        # Get the mean value of the pixels in the diagonal, form 0,0 to h,w 
        # and from h,0 to 0,w
        coordinates = {'y1_1': 0, 'x1_1': 0, 'y2_1': h, 'x2_1': w, 'y1_2': h, 'x1_2': 0, 'y2_2': 0, 'x2_2': w}
        for i in range(2):
            d = []
            y1_aux, x1_aux = 0, 0
            y2_aux, x2_aux = h, w
            for y, x in zip(y_coords[i], x_coords[i]):
                d.append(np.mean(image[y, x, :]))

            # Get the location of the first point where the threshold is crossed
            for idx, value in enumerate(d):
                if value >= threshold:
                    coordinates['y1_' + str(i + 1)] = y_coords[i][idx]
                    coordinates['x1_' + str(i + 1)] = x_coords[i][idx]
                    break

            # Get the location of the last point where the threshold is crossed
            for idx, value in enumerate(reversed(d)):
                if value >= threshold:
                    coordinates['y2_' + str(i + 1)] = y_coords[i][-idx if idx != 0 else -1]
                    coordinates['x2_' + str(i + 1)] = x_coords[i][-idx if idx != 0 else -1]
                    break

        # Set the coordinates to crop the image
        y1 = max(coordinates['y1_1'], coordinates['y2_2'])
        y2 = min(coordinates['y2_1'], coordinates['y1_2'])
        x1 = max(coordinates['x1_1'], coordinates['x1_2'])
        x2 = min(coordinates['x2_1'], coordinates['x2_2'])
       
        if (y2 < y1):
            y1 = coordinates['y2_2']
            y2 = coordinates['y2_1']
        
        if (x2 < x1):
            x1 = coordinates['x1_1']
            x2 = coordinates['x2_2']

        image_cropped = image[y1:y2, x1:x2, :]
        # image_cropped_size = image_cropped.shape

        return image_cropped  # , image_cropped_size

    def get_seg_mask(self, image: np.ndarray):
        """
        Obtain segmented lesion mask from image. Uses thresholding method from R channel.
        It filters the largest connected component

        Args:
            image (np.ndarray): Image to segment

        Returns:
            mask (np.ndarray): Binary mask with the segmented lesion
        """
        # obtain segmentation with thresholding method

        r_norm = image[:,:,2]*(1/np.sqrt(np.sum(image.astype(np.float32)**2 + np.finfo(float).eps, axis=-1)))
        rnormg = (cv2.GaussianBlur(r_norm, ksize = (0,0), sigmaX=3, borderType = cv2.BORDER_DEFAULT)*255).astype(np.uint8)
        _,mask_r = cv2.threshold(rnormg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Filling holes
        contour_r,_ = cv2.findContours(mask_r,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour_r:
            cv2.drawContours(mask_r,[cnt],0,255,-1)

        # filter largest connected component
        connectivity = 4  
        output = cv2.connectedComponentsWithStats(mask_r, connectivity, cv2.CV_32S)
        
        # connected components
        labels = output[1]
        # statistics matrix
        stats = output[2]

        # get maximum maximum size not considering the background

        bkgd_bbox = [0, 0, image.shape[1], image.shape[0]]
        cc_areas = {lab:stats[lab,4] for lab, row in enumerate(stats[:,:4]) if (row != bkgd_bbox).any()}
        
        if not not cc_areas:
            top_lab = max(cc_areas, key=cc_areas.get)
            final_mask = np.uint8(labels == top_lab)
        else:
            return mask_r
        # discard filtered mask if background do not follow the set condition
        if mask_r.sum() > (1000*final_mask.sum()):
            return mask_r
        
        return final_mask  