import time
import numpy as np
from detection.shared_functions import convert_color, color_hist, get_hog_features, feature_vector, bin_spatial
from sklearn.model_selection import train_test_split
from detection.data import load_data, save_pickle_file
from detection.config import DetectionConfig
from sklearn.preprocessing import StandardScaler

def extract_features(imgs, config=DetectionConfig()):
  """
  extract features from a list of images
  :param imgs: Array
  :param config: DetectionConfig, config values to use
  :returns: Array of array of features
  """
	print("extracting features from", len(imgs), "images")
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for img in imgs:
		features.append(extract_feature(img, config=config))
	# Return list of feature vectors
	print("done extracting features from", len(imgs), "images")
	return features 

def extract_feature(img, config=DetectionConfig()):
  """
  extract features from a single image
  :param img: Array
  :param config: DetectionConfig, config values to use
  :returns: Array of features
  """	
	cspace_img = convert_color(img, 'RGB2' + config.cspace)/255
	hog_img = convert_color(img, 'RGB2' + config.hog_cspace)/255
	hist_features = color_hist(cspace_img, nbins=config.hist_bins)
	binned_features = bin_spatial(cspace_img)
	hog0 = get_hog_features(hog_img[:,:,0], config.orient, config.pix_per_cell, config.cell_per_block, transform_sqrt=config.transform_sqrt)
	hog1 = get_hog_features(hog_img[:,:,1], config.orient, config.pix_per_cell, config.cell_per_block, transform_sqrt=config.transform_sqrt)
	hog2 = get_hog_features(hog_img[:,:,2], config.orient, config.pix_per_cell, config.cell_per_block, transform_sqrt=config.transform_sqrt)

	return feature_vector(hog0, hog1, hog2, hist_features, binned_features)

def test_train_data(X_scaler, rand_state, config=DetectionConfig()):
  """
  loads data and then splits it into test and train. Also fits the scaler
  :param X_scaler: the scaler to use. Un-fit
	:param rand_state: the random state to use
  :param config: DetectionConfig, config values to use
  :returns: X_train, X_test, y_train, y_test image sets
  """	
	t = time.time()
	# load data
	vehicles_images, non_vehicles_images = load_data(config.size)
	# extract features
	features_vehicles = extract_features(vehicles_images, config=config)
	features_non_vehicles = extract_features(non_vehicles_images, config=config)
	# Create an array stack of feature vectors
	X = np.vstack((features_vehicles, features_non_vehicles)).astype(np.float64) 
	# Fit a per-column scaler
	X_scaler.fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)
	# Define the labels vector
	y = np.hstack((np.ones(len(features_vehicles)), np.zeros(len(features_non_vehicles))))
	# split data into test and train
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

	print("Took {} seconds to extract {} features.".format(round(time.time()-t, 2), len(features_vehicles) + len(features_non_vehicles)))
	return X_train, X_test, y_train, y_test

def train_classifier(X_train, X_test, y_train, y_test, X_scaler, rand_state, config=DetectionConfig()):
  """
  tran a classifier
  :param X_scaler: the scaler to use. Un-fit
	:param rand_state: the random state to use
  :param config: DetectionConfig, config values to use
  :returns: X_train, X_test, y_train, y_test image sets
  """	
	clf = config.classifier(rand_state)

	# Check the training time for the SVC
	print("Training classifier")
	t=time.time()
	clf.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'seconds to train', config.classifier_name, '...')
	# Check the score of the SVC
	# Check the prediction time for a single sample
	t=time.time()
	n_predict_test = len(X_test)
	n_predict_train = len(X_train)
	print('Test Accuracy of', config.classifier_name, '=', round(clf.score(X_test, y_test), 4))
	print('Train Accuracy of', config.classifier_name, '=', round(clf.score(X_train, y_train), 4))
	t2 = time.time()
	print(round(t2-t, 2), 'seconds to predict', n_predict_test,'test labels and', n_predict_train, 'train labels with', config.classifier_name)

	print("Saving classifier.")
	save_pickle_file(config.classifier_file(), clf)
	print("Done saving classifier.")
	print("Saving scaler.")
	save_pickle_file(config.scaler_file(), X_scaler)
	print("Done saving scaler.")




