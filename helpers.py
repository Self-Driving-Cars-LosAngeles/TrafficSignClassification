import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.utils import shuffle
from sklearn import preprocessing
import cv2
import random

def normalize_grayscale(image_data):
    a = np.float32(0.05)
    b = np.float32(0.95)
    image_data_shape = image_data.shape
    gray_data = np.zeros(image_data.shape[:-1])
    for idx, image in enumerate(image_data):
        gray_data[idx] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    x_max = np.max(gray_data, axis = 0)
    x_min = np.min(gray_data, axis = 0)
    gray_data_maxmin = a + (gray_data - x_min)*(b-a)/(x_max - x_min)
    return gray_data_maxmin



def normal_equalize(image):
    """
    This function utilizes OpenCV's equalizeHist function.
    input: an image (RGB)
    It then converts the image from RGB to YUV and performs histogram equalization on the Y (Luminance) Parameter
    Then it rewrites the Y layer of the input image to the equalized Y layer, then converts back to RGB
    output: an image (RGB) with same shape
    """
    image_eq = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image_eq[:,:,0] = cv2.equalizeHist(image_eq[:,:,0])
    return cv2.cvtColor(image_eq, cv2.COLOR_YUV2RGB) # Back to RGB

def histogram_equalize_data(image_data):
    """
    This function takes in an image dataset and returns an histogram equalized dataset.
    It utilizes the normal_equalize function which uses OpenCVs equalizeHist function
    """
    fill_data = np.zeros(image_data.shape)
    for idx, image in enumerate(image_data):
        fill_data[idx] = normal_equalize(image)
    return fill_data

def transform_image(img, ang_range, shear_range, trans_range):
    """
    This function takes in an image as input and performs angle changing, shearing, 
    and translation (all affine warping techniques) on the original image and 
    returns a warped image with the same dimensions.
    ang_range: range of angles for rotation
    shear_range: range of values to apply affine transformation to
    trans_range: range of values to apply translation to
    
    To randomly scale the affine parameters we apply multiply each parameter by a 
    uniform distribution constant. (Uniform because we seek to give equal probability to each rotation)
    A random uniform distribution is used to generate different parameters for transformation
    """

    cols, rows = img.shape[:2]
    
    # Rotation:
    
    ang_rot = ang_range * np.random.uniform() - ang_range / 2
    
    # We need to get a rotation matrix and then apply it to warpAffine
    # cv2.getRotationMatrix2D(center, angle, scale)
    # center = rows/2, columns/2
    # angle = ang_rot
    # scale = 1
    Rot_M = cv2.getRotationMatrix2D((rows/2, cols/2), ang_rot, 1)
    
    
    
    # Translation:
    
    # Note: np.random.uniform() is a number [0, 1] which differs from
    # np.random.uniform(trans_range), which is a number [0, trans_range] because we are 
    # ultimately trying to scale that value, instead of generating a value from 1 to 20
    # scale it and subtract it from half the trans_range so it is either positive or negative
    # meaning we can translate in either direction
    
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    
    # Shearing:
    
    pts1 = np.float32([[5, 5],[20, 5],[5, 20]])
    
    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
    
    pts2 = np.float32([[pt1, 5],[pt2, pt1],[5, pt2]])
    
    Shear_M = cv2.getAffineTransform(pts1, pts2)
    
    # --------------------Apply warping------------------
    # cv2.warpAffine(src, M, dsize):
    # src: image to warp
    # M: 2x3 transformation matrix **Check size* 
    # dsize: size of output image (in this case the same as input image)
    
    # Warp rotation:
    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    
    # Warp translation: 
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    
    # Warp shear: 
    img = cv2.warpAffine(img, Shear_M, (cols, rows))
    
    return img

def copy_and_transform_dataset(x, y):
    # print(cache_unique_class_index.values())
    cache_unique_class_index.items()
    for label, index in cache_unique_class_index.items():
    #     print('label ', label)
    #     print('index: ', index)
        scaling_factor = data_pd_sorted.loc[data_pd_sorted['ClassId'] == y_train[index[0]]]['Scaling_Factor'].values[0]
        scaling_factor = scaling_factor.astype(np.uint8)
        img = X_train_equalized[index[0]]
        for i in range(scaling_factor):
            x.append(transform_image(img,20,10,5))
            y.append(label)
    return (np.asarray(x), np.asarray(y))

def transform_image_data(image_data, ang_range, shear_range, trans_range):
    """
    In this piece we seek to transform a dataset of images and return the new dataset
    """
    # create blank_fill
    fill_data = np.zeros(image_data.shape)
    for idx, image in enumerate(image_data):
        fill_data[idx] = transform_image(image, ang_range, shear_range, trans_range)
    return fill_data

