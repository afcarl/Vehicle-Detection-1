#!/bin/python

class DetectionConfig:
	def __init__(self, size="full", n_estimators=125, learning_rate=1.0, cspace='RGB', hist_bins=32, orient=9, pix_per_cell=16, cell_per_block=2, transform_sqrt=False):
		self.size = size
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.cspace = cspace
		self.hist_bins = hist_bins
		self.orient = orient
		self.pix_per_cell = pix_per_cell
		self.cell_per_block = cell_per_block
		self.transform_sqrt = transform_sqrt

	def tag_name(self):
		return "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.size, self.n_estimators, self.learning_rate, self.cspace, self.hist_bins, self.orient, self.pix_per_cell, self.cell_per_block, self.transform_sqrt)

	def classifier_file(self, classifier="AdaBoost"):
		return "classifier_" + classifier + "_" + self.tag_name() + ".p"

	def scaler_file(self):
		return "scaler_" + self.tag_name() + ".p"