import cv2 as cv
import numpy as np
import os
from skimage.transform import resize


def sum_operation_se_different_directions(img,operation, width,height,n_se):
    # create SEs
    base = np.zeros([width, width])
    k = int(width / 2 - height / 2)
    while k <= (width / 2 + height / 2):
        base = cv.line(base, (0, k), (width, k), 255)
        k = k + 1
        #print(k)
    SEs = []
    SEs.append(base)
    angle = 180.0 / n_se
    for k in range(1, n_se):
        SEs.append(cv.warpAffine(base, cv.getRotationMatrix2D((base.shape[0] / 2, base.shape[1] / 2), k * angle, 1.0),(width, width)))
    #cv.imshow("see",SEs[0])
    #print(SEs[2].shape)
    open_sum = np.uint16(0*cv.morphologyEx(img, operation, np.uint8(SEs[0])))
    for se in SEs:
        open_sum += cv.morphologyEx(img, operation, np.uint8(se))
    result= cv.normalize(open_sum, 0, 255, norm_type=cv.NORM_MINMAX)
    return np.uint8(result)


def smplextract_hair(src):

#17 17 10
    
# Convert the original image to grayscale
    
    img_gray=cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    contrast_enhanced_gray_img = clahe.apply(img_gray)

    width= 29
    height=2
    n_se=15
    
    #cv.imshow("original_image",src)
    sum_black_hats=sum_operation_se_different_directions(contrast_enhanced_gray_img,cv.MORPH_BLACKHAT, width,height,n_se)
    #cv.imshow("sum_blackhats", sum_black_hats)
    ret, bin_img = cv.threshold(sum_black_hats, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #cv.imshow("Thresholded Mask",bin_img)
    #cv.imwrite('thresholded_sample1.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    dilated_bin_img=cv.dilate(bin_img,cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)))
    #cv.imshow("dilated", dilated_bin_img)
    # inpaint the original image depending on the mask
    dst = cv.inpaint(src,dilated_bin_img,7,cv.INPAINT_TELEA)
    #cv.imshow("Inpaint",dst)
    #cv.waitKey(50)
    return dst

# img = cv.imread('D:/Respaldo/Documents/MAIA/UNICAS/AImageP/project/FILES/Dataset/train/ISIC_0000043.jpg')  
# result= smplextract_hair(img)
# cv.imshow("result",result)
# cv.waitKey(0)