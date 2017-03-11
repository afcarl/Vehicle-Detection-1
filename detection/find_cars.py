#!/bin/python

import numpy as np
import cv2
from detection.shared_functions import get_hog_features, bin_spatial, color_hist, convert_color
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import time

# for testing
# return 1 once per frequency i.e. higher number is less often
import random
class RandomCLF:
  def __init__(self, frequency=100):
    self.frequency = frequency

  def predict(self, features):
    if random.randint(0, self.frequency) >= self.frequency - 1:
      return 1
    return 0

class CarFinder:
  def __init__(self, clf=RandomCLF(), scaler=None):
    self.clf = clf
    self.scaler = scaler # TODO
  	
  def find_cars(self, image, ystart = 400, ystop=700, scale=1.5, orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32):
    X_scaler = StandardScaler()
    return find_cars(image, ystart, ystop, scale, self.clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
  img = img.astype(np.float32)/255

  img_tosearch = img[ystart:ystop,:,:]
  ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
  if scale != 1:
    imshape = ctrans_tosearch.shape
    ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
      
  ch1 = ctrans_tosearch[:,:,0]
  ch2 = ctrans_tosearch[:,:,1]
  ch3 = ctrans_tosearch[:,:,2]

  # Define blocks and steps as above
  nxblocks = (ch1.shape[1] // pix_per_cell)-1
  nyblocks = (ch1.shape[0] // pix_per_cell)-1 
  nfeat_per_block = orient*cell_per_block**2
  # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
  window = 64
  nblocks_per_window = (window // pix_per_cell)-1 
  cells_per_step = 2  # Instead of overlap, define how many cells to step
  nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
  nysteps = (nyblocks - nblocks_per_window) // cells_per_step

  # Compute individual channel HOG features for the entire image
  hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
  hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
  hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

  bboxes = []

  for xb in range(nxsteps):
    for yb in range(nysteps):
      ypos = yb*cells_per_step
      xpos = xb*cells_per_step
      # Extract HOG for this patch
      hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
      hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
      hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
      hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

      xleft = xpos*pix_per_cell
      ytop = ypos*pix_per_cell

      # Extract the image patch
      subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

      # Get color features
      spatial_features = bin_spatial(subimg, size=spatial_size)
      hist_features = color_hist(subimg, nbins=hist_bins)

      # TODO remove fit
      # Scale features and make a prediction
      test_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
      test_features = X_scaler.fit_transform(test_features)
      
      test_prediction = clf.predict(test_features)

      if test_prediction == 1:
        xbox_left = np.int(xleft*scale)
        ytop_draw = np.int(ytop*scale)
        win_draw = np.int(window*scale)
        points = (xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)
        bboxes.append(points)
                
  return bboxes
