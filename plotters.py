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


def plot_histograms(image, image_equalized, image_clahe):
    plt.figure(figsize = (11,8))
    
    # Image
    hist_image,bins_image = np.histogram(image.flatten(),256,[0,256])
    cdf_image = hist_image.cumsum()
    cdf_normalized_image = cdf_image * hist_image.max()/ cdf_image.max()
    
    # Normal Equalization
    hist_image_equalized, bins_image_equalized = np.histogram(image_equalized.flatten(), 250, [0, 256])
    cdf_image_equalized = hist_image_equalized.cumsum()
    cdf_normalized_image_equalized = cdf_image_equalized * hist_image_equalized.max() / cdf_image_equalized.max()
    
    # CLAHE equalization
    hist_image_clahe, bins_image_clahe = np.histogram(image_clahe.flatten(), 250, [0, 256])
    cdf_image_clahe = hist_image_clahe.cumsum()
    cdf_normalized_image_clahe = cdf_image_clahe * hist_image_clahe.max() / cdf_image_clahe.max()
    
    # Original image plot
    plt.subplot(321)
    plt.title('image')
    plt.imshow(image)

    # Original image histogram
    plt.subplot(322)
    plt.plot(cdf_normalized_image, color = 'b')
    plt.title('Original Histogram')
    plt.hist(image.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    
    # Equalized image plot
    plt.subplot(323)
    plt.title('normal_equalized')
    plt.imshow(image_equalized)

    # Equalized image histogram
    plt.subplot(324)
    plt.plot(cdf_normalized_image_equalized, color = 'b')
    plt.title('Equalized Histogram')
    plt.hist(image_equalized.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    
    # CLAHE image plot
    plt.subplot(325)
    plt.title('Clahe Equalized')
    plt.imshow(image_clahe)
    
    # Clahe Histogram
    plt.subplot(326)
    plt.title('Clahe Histogram')
    plt.xlabel('Brightness')
    plt.ylabel('Occurance')
    plt.plot(cdf_normalized_image_clahe, color = 'b')
    plt.hist(image_clahe.flatten(), 256, [0, 256], color = 'r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc = 'upper left')
    
    
    
plt.show()


def plot_rand_original_grayscale_normalized(n_row, n_col,name,  original, grayscaled):
    plt.figure(figsize = (11,8))
    
    plt.suptitle(name, fontsize = "12")
    plt.subplot(121)
    plt.axis('off')
    plt.title('Original')
    plt.imshow(original)
    
    plt.subplot(122)
    plt.axis('off')
    plt.title('Grayscaled')
    plt.imshow(grayscaled, cmap="gray")
    
plt.show()
