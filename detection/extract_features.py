import time
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from detection.shared_functions import convert_color, color_hist, get_hog_features
from sklearn.model_selection import train_test_split
from detection.data import load_data, save_pickle_file
from detection.config import DetectionConfig
from sklearn.preprocessing import StandardScaler

def extract_features(imgs, config=DetectionConfig()):
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
	img = convert_color(img, 'RGB2'+ config.cspace)
	hist_features = color_hist(img, nbins=config.hist_bins)
	hog0 = get_hog_features(img[:,:,0], config.orient, config.pix_per_cell, config.cell_per_block, transform_sqrt=config.transform_sqrt)
	hog1 = get_hog_features(img[:,:,1], config.orient, config.pix_per_cell, config.cell_per_block, transform_sqrt=config.transform_sqrt)
	hog2 = get_hog_features(img[:,:,2], config.orient, config.pix_per_cell, config.cell_per_block, transform_sqrt=config.transform_sqrt)
	hogs = np.hstack((hog0, hog1, hog2))
	return np.concatenate((hist_features, hogs))

def train_classifier(config=DetectionConfig()):
	classifier_name = "AdaBoost"
	# load data
	vehicles_images, non_vehicles_images = load_data(config.size)
	# extract features
	features_vehicles = extract_features(vehicles_images, config=config)
	features_non_vehicles = extract_features(non_vehicles_images, config=config)
	# Create an array stack of feature vectors
	X = np.vstack((features_vehicles, features_non_vehicles)).astype(np.float64) 
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)
	# Define the labels vector
	y = np.hstack((np.ones(len(features_vehicles)), np.zeros(len(features_non_vehicles))))
	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)

	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
	clf = AdaBoostClassifier(base_estimator=None, n_estimators=config.n_estimators, learning_rate=config.learning_rate, random_state=rand_state)

	# Check the training time for the SVC
	print("About to train classifier")
	t=time.time()
	clf.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train', classifier_name, '...')
	# Check the score of the SVC
	# Check the prediction time for a single sample
	t=time.time()
	n_predict_test = len(X_test)
	n_predict_train = len(X_train)
	print('Test Accuracy of', classifier_name, '=', round(clf.score(X_test, y_test), 4))
	print('Train Accuracy of', classifier_name, '=', round(clf.score(X_train, y_train), 4))
	t2 = time.time()
	print(round(t2-t, 5), 'Seconds to predict', n_predict_test,'test labels and', n_predict_train, 'test labels with', classifier_name)

	print("Saving classifier")
	save_pickle_file(config.classifier_file(classifier_name), clf)
	print("Done saving classifier")
	print("saving scaler")
	save_pickle_file(config.scaler_file(), X_scaler)
	print("done saving scaler")




