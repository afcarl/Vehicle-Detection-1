#!/bin/python 

import numpy as np
import cv2
from detection.find_cars import CarFinder
from scipy.ndimage.measurements import label

class Road:
  def __init__(self):
    self.frame_count = 0
    self.car_finder = CarFinder()

  def process_image(self, image):
    self.frame_count+=1

    bboxes = self.car_finder.find_cars(image)
    labels = bboxes_to_labels(np.zeros_like(image[:,:,0]), bboxes)
    hot_boxes = labels_to_bboxes(labels)
    initial_boxes_img = draw_boxes(image, bboxes, color=(0,255,0))
    return draw_boxes(initial_boxes_img, hot_boxes)

# draw boxes on copy of original image
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
  # Make a copy of the image
  draw_img = np.copy(img)
  # Iterate through the bounding boxes
  for bbox in bboxes:
    # Draw a rectangle given bbox coordinates
    cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
  # Return the image copy with boxes drawn
  return draw_img

# convert the potentially overlapping bound boxes from the search to labels using heatmap
def bboxes_to_labels(heatmap, bboxes, threshold=1):
	for bbox in bboxes:
    # Add += 1 for all pixels inside each bbox
    # Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	heatmap = np.clip(heatmap, 0, 255)
	return label(heatmap)

# convert heatmap labels to single set of bounding boxes
def labels_to_bboxes(labels):
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
