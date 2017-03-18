#!/bin/python

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
import random

class DetectionConfig:
	"""
  Class containing all config imformation. 

  Helpful for loading and saving classifier and scaler with exact classifier configuration

  :param classifier_name: type of classifier
  :param size: dataset to load, `small` or `full`
  :param n_estimators: number of estimators for adaboost
  :param learning_rate: learning_rate for adaboost
  :param cspace: color space for color histogram
  :param hog_cspace: color space for hog function
  :param hist_bins: number of histogram bins
  :param orient: number of orientations for hog function
  :param pix_per_cell: number of pixels per cell of hog function
  :param cell_per_block: cells per block in hog function
  :param transform_sqrt: whether to transform the image in the hog functions. Helpful for controlling for shadows
  """
	def __init__(self, classifier_name="LinearSVC", size="full", n_estimators=125, learning_rate=1.0, cspace='RGB', hog_cspace='YCrCb', hist_bins=32, orient=8, pix_per_cell=16, cell_per_block=2, transform_sqrt=True):
		self.classifier_name = classifier_name
		self.size = size
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.cspace = cspace
		self.hog_cspace = hog_cspace
		self.hist_bins = hist_bins
		self.orient = orient
		self.pix_per_cell = pix_per_cell
		self.cell_per_block = cell_per_block
		self.transform_sqrt = transform_sqrt

	def tag_name(self):
		"""
	  The generalized tag string for the classifier and scaler
		:returns: String, underscored list of values in the config.
	  """
		return '_'.join([self.size, str(self.n_estimators), str(self.learning_rate), self.cspace, self.hog_cspace, str(self.hist_bins), str(self.orient), str(self.pix_per_cell), str(self.cell_per_block), str(self.transform_sqrt)])

	def classifier(self, rand_state):
		"""
	  The classifier to be trained. 

	  Classifiers in scikit-learn all have the same 4 functions so it doesnt 

	  :param rand_state: Double, the random state to be used. 
		:returns: Classifier, dependends on the classifier_name
	  """
		if self.classifier_name is "AdaBoost":
			return AdaBoostClassifier(base_estimator=None, n_estimators=self.n_estimators, learning_rate=self.learning_rate, random_state=rand_state)
		elif self.classifier_name is "LinearSVC":
			return LinearSVC(random_state=rand_state, tol=0.001, dual=False)
		elif self.classifier_name is "Random":
			return RandomCLF()
		else:
			return None

	def classifier_file(self):
		"""
	  The classifier pickle file name. 
		:returns: String
	  """
		return "classifier_" + self.classifier_name + "_" + self.tag_name() + ".p"

	def scaler_file(self):
		"""
	  The scaler pickle file name. 
		:returns: String
	  """
		return "scaler_" + self.tag_name() + ".p"

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
