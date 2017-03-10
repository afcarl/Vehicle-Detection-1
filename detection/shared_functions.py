#!/bin/python 

import cv2
import numpy as np
from skimage.feature import hog

# Define a function to compute color histogram features  
def color_hist(img, nbins=32):
  # Compute the histogram of the color channels separately
  channel1_hist = np.histogram(img[:,:,0], bins=nbins)
  channel2_hist = np.histogram(img[:,:,1], bins=nbins)
  channel3_hist = np.histogram(img[:,:,2], bins=nbins)
  # Concatenate the histograms into a single feature vector
  hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
  return hist_features

def convert_color(img, conv='RGB2YCrCb'):
  if conv == 'RGB2RGB':
    return np.copy(img) 
  if conv == 'RGB2YCrCb':
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
  if conv == 'BGR2YCrCb':
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
  if conv == 'RGB2LUV':
    return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
  if conv == 'RGB2HSV':
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, transform_sqrt=False, vis=False, feature_vec=True):
  if vis == True:
    features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=transform_sqrt, visualise=True, 
      feature_vector=False)

    return features, hog_image

  else:      
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=transform_sqrt, 
      visualise=False, feature_vector=feature_vec)
    
    return features


# # Define a function to compute color histogram features  
# # Pass the color_space flag as 3-letter all caps string
# # like 'HSV' or 'LUV' etc.
# def bin_spatial(img, color_space='RGB', size=(32, 32)):          
#     # Use cv2.resize().ravel() to create the feature vector
#     return cv2.resize(img, size).ravel() 
