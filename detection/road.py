#!/bin/python 

import numpy as np
import cv2
from detection.find_cars import CarFinder
from scipy.ndimage.measurements import label
from detection.config import DetectionConfig
from detection.data import open_pickle_file


class Road:
  """
  Overarching class which magages state while processing a video
  :param frame_count: Int, the number of frames seen, also the current frame.
  :param car_finder: CarFinder, the class responsible for identifiying the cars in the image
  :param verbose: Bool, whether to use verbose logging or not
  :param recent_bboxes: Array, the last n number of boxes identified in the image
  :param bbox_size: Int, number of boxes to keep while processing image
  :param threshold: Double, The threshold multiplier for the heat map. bbox_size * threshold = heatmat_threshold
  :param full_heatmap: Array, the most recent heatmap before the thresholding. For debugging/visualizing
  :param threshold_heatmap: Array, the most recent heatmap after thresholding
  :param final_bboxes: Array, the boxes which will be drawn on the image
  """
  def __init__(self, config=DetectionConfig(), bbox_size=8, threshold=4.2, verbose=False):
    clf = open_pickle_file(config.classifier_file())
    scaler = open_pickle_file(config.scaler_file())
    self.frame_count = 0
    self.car_finder = CarFinder(clf=clf, scaler=scaler)
    self.verbose = verbose
    self.recent_bboxes = []
    self.bbox_size = bbox_size
    self.threshold = threshold

    # for debug save most recent heatmap
    self.full_heatmap = None
    self.threshold_heatmap = None
    self.final_bboxes = None

  def process_image(self, image):
    """
    Function to pass to movie.editor.VideoFileClip
    :param image: Array, an image to search for cars in
    :returns: image of same size with boxes drawn
    """
    self.frame_count+=1

    search_areas = [
    (375, 550, 1.0)
    , (375, 650, 2.0)
    # , (350, 650, 3.0)
    # , (350, 650, 4.0)
    # , (350, 650, 5.0)
    # , (375, 550, 0.5)
    # , (375, 500, 0.75)
    ]

    bboxes = []
    success = []
    for area in search_areas:
      boxes = self.car_finder.find_cars(image, ystart=area[0], ystop=area[1], scale=area[2])
      if self.verbose:
        print("area:", area[0:2], "scale:", area[2])
      if len(boxes) != 0:
        if self.verbose:
          print("found", len(boxes), "boxes!")
        success.append((area, len(boxes)))
        for box in boxes: bboxes.append(box)

    if self.verbose: print(success)

    self.recent_bboxes.append(bboxes)
    if len(self.recent_bboxes) > self.bbox_size:
      self.recent_bboxes.pop(0)

    labels = self.bboxes_to_labels(np.zeros_like(image[:,:,0]))
    hot_boxes = self.labels_to_bboxes(labels)
    self.final_bboxes = hot_boxes
    return self.draw_boxes(image, hot_boxes)

  def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
    """
    draw boxes on copy of original image
    :param img: Array, an image to draw on
    :param bboxes: Array, list of points to draw a rectangle from
    :param color: Tuple, rgb color
    :param thick: Int, thickness of boxes
    :returns: image of same size with boxes drawn
    """

    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
      # Draw a rectangle given bbox coordinates
      cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), color, thick)
    # Return the image copy with boxes drawn
    return draw_img

  def bboxes_to_labels(self, heatmap):
    """
    convert the potentially overlapping bound boxes from the search to labels using heatmap
    :param heatmap: Array, an array of zeros matching the shape[0:2] of the image being processed
    :returns: heatmap after thresholding
    """
    for bboxes in self.recent_bboxes:
      for bbox in bboxes:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

    self.full_heatmap = np.copy(heatmap)
    # Zero out pixels below the threshold
    heatmap[double(heatmap) <= self.bbox_size * self.threshold] = 0
    heatmap = np.clip(heatmap, 0, 255)
    self.threshold_heatmap = np.copy(heatmap)
    return label(heatmap)

  def labels_to_bboxes(self, labels):
    """
    convert heatmap labels to single set of bounding boxes
    :param labels: Array, list of bounding boxes labels in the image
    :returns: array of bounding boxes to draw
    """
    result = []
    for car_number in range(1, labels[1]+1):
      # Find pixels with each car_number label value
      nonzero = (labels[0] == car_number).nonzero()
      # Identify x and y values of those pixels
      nonzeroy = np.array(nonzero[0])
      nonzerox = np.array(nonzero[1])
      # Define a bounding box based on min/max x and y
      result.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))
    return result
