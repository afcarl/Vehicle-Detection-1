#!/bin/python

import numpy as np
import cv2
from detection.shared_functions import get_hog_features, color_hist, convert_color, feature_vector, bin_spatial
from sklearn.preprocessing import StandardScaler
from detection.config import DetectionConfig
import matplotlib.pyplot as plt
import random

class RandomCLF:
  """
  A "classifier" which randomly assigns returns true. Helpful when testing drawing bounding boxes
  :param frequency: Int, return 1 once per frequency i.e. higher number is less often
  """
  def __init__(self, frequency=100):
    self.frequency = frequency

  def predict(self, features):
    """
    Matching classifier function signiture. Returns 1 or 0.
    :param features: doesnt matter.
    """
    if random.randint(0, self.frequency) >= self.frequency - 1:
      return 1
    return 0

class CarFinder:
  """
  Responsible for finding cars using a classifier
  :param clf: Classifier, a trained classifier
  :param scaler: Scaler, a trained scaler
  :param config: DetectionConfig, the config params
  :param verbose: Bool, whether to turn on verbose logging
  """
  def __init__(self, clf=RandomCLF(), scaler=None, config=DetectionConfig(), verbose=False):
    self.clf = clf
    self.scaler = scaler
    self.config = config
    self.verbose = verbose
  	
  def find_cars(self, image, ystart=400, ystop=650, scale=1.5):
    """
    find cars in image using a classifier
    :param image: the image to search
    :param ystart: Int, the top of the image to search from
    :param ystop: Int, the bottom of the image to search from
    :param scale: Double, the scale factor to apply to the image. 
      scale > 1 means shrink the image and draw larger boxes
    :returns: Array of bounding boxes
    """
    return find_cars(image, ystart, ystop, scale, self.clf, self.scaler, self.config, self.verbose)

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, clf, scaler, config, verbose):
  """
  find cars in image using a classifier.

  Passes a color histogram, a reduced image, and hog features to the classifier. The hog features
  are calculated once for the entire image and then the image is scanned using a sliding window
  to pass the features to the classifier. Points passing the classifier are collected and returned. 

  :param img: the image to search
  :param ystart: Int, the top of the image to search from
  :param ystop: Int, the bottom of the image to search from
  :param scale: Double, the scale factor to apply to the image. 
    scale > 1 means shrink the image and draw larger boxes
  :param clf: Classifier, a trained classifier
  :param scaler: Scaler, a trained scaler
  :param config: DetectionConfig, the config params
  :param verbose: Bool, whether to turn on verbose logging
  :returns: Array of bounding boxes
  """
  img_tosearch = convert_color(img, 'RGB2'+ config.cspace)[ystart:ystop,:,:]/255
  img_tosearch_hog = convert_color(img, 'RGB2'+ config.hog_cspace)[ystart:ystop,:,:]/255

  if scale != 1:
    imshape = img_tosearch.shape
    img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    img_tosearch_hog = cv2.resize(img_tosearch_hog, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

  if verbose:
    plt.imshow(img_tosearch)
    plt.show()
      
  ch1 = img_tosearch_hog[:,:,0]
  ch2 = img_tosearch_hog[:,:,1]
  ch3 = img_tosearch_hog[:,:,2]

  # Define blocks and steps as above
  nxblocks = (ch1.shape[1] // config.pix_per_cell)-1
  nyblocks = (ch1.shape[0] // config.pix_per_cell)-1 
  nfeat_per_block = config.orient*config.cell_per_block**2
  # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
  window = 64
  nblocks_per_window = (window // config.pix_per_cell)-1 
  cells_per_step = 1  # Instead of overlap, define how many cells to step
  nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
  nysteps = (nyblocks - nblocks_per_window) // cells_per_step

  # Compute individual channel HOG features for the entire image
  hog0 = get_hog_features(ch1, config.orient, config.pix_per_cell, config.cell_per_block, feature_vec=False)
  hog1 = get_hog_features(ch2, config.orient, config.pix_per_cell, config.cell_per_block, feature_vec=False)
  hog2 = get_hog_features(ch3, config.orient, config.pix_per_cell, config.cell_per_block, feature_vec=False)

  bboxes = []

  for xb in range(nxsteps):
    for yb in range(nysteps):
      ypos = yb*cells_per_step
      xpos = xb*cells_per_step
      # Extract HOG for this patch
      hog_feat0 = hog0[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
      hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
      hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 

      xleft = xpos*config.pix_per_cell
      ytop = ypos*config.pix_per_cell

      # Extract the image patch
      subimg = cv2.resize(img_tosearch[ytop:ytop+window, xleft:xleft+window], (window,window))

      # Get color features
      hist_features = color_hist(subimg, nbins=config.hist_bins)

      # Get binned color features
      binned_features = bin_spatial(subimg)

      # Scale features and make a prediction
      test_features = feature_vector(hog_feat0, hog_feat1, hog_feat2, hist_features, binned_features).reshape(1,-1) 
      test_features = scaler.transform(test_features)
      
      test_prediction = clf.predict(test_features)

      if test_prediction == 1:
        if verbose:
          plt.imshow(subimg)
          plt.show()

        xbox_left = np.int(xleft*scale)
        ytop_draw = np.int(ytop*scale)
        win_draw = np.int(window*scale)
        points = (xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)
        bboxes.append(points)
                
  return np.array(bboxes)
