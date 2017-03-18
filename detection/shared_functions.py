#!/bin/python 

import cv2
import numpy as np
from skimage.feature import hog

def bin_spatial(img, size=(32, 32)):
  """
  compute binned color features of image
  :param img: Array
  :param size: tuple, size to scale the image to
  :returns: Array
  """
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):
  """
  compute color histogram features by image channel 
  :param img: Array
  :param nbins: Int, number of histogram bins
  :returns: Array
  """
  # Compute the histogram of the color channels separately
  channel1_hist = np.histogram(img[:,:,0], bins=nbins)
  channel2_hist = np.histogram(img[:,:,1], bins=nbins)
  channel3_hist = np.histogram(img[:,:,2], bins=nbins)
  # Concatenate the histograms into a single feature vector
  hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
  return hist_features

def convert_color(img, conv='RGB2YCrCb'):
  """
  convert image from one color space to another
  :param img: Array
  :param conv: String, color space
  :returns: image
  """
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
  """
  return HOG features and maybe visualization
  :param img: Array
  :param orient: number of orientations for hog function
  :param pix_per_cell: number of pixels per cell of hog function
  :param cell_per_block: cells per block in hog function
  :param transform_sqrt: whether to transform the image in the hog functions. Helpful for controlling for shadows
  :param vis: Bool, visualize the hog result or not
  :param feature_vec: Bool, return the feature vector ravel-ed or not
  :returns: features and maybe hog image
  """
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

def feature_vector(hog0, hog1, hog2, color_hist, bin_spatial):
  hogs = np.hstack((hog0, hog1, hog2))
  return np.concatenate((bin_spatial, color_hist, hogs))
