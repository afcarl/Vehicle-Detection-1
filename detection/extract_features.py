import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

# AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)

def extract_features(imgs, cspace='RGB', hist_bins=32):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in imgs:
        features.append(extract_feature(img, cspace="HSV", hist_bins=hist_bins))
    # Return list of feature vectors
    return features 

def extract_feature(img, cspace='RGB', hist_bins=32, orient=9, pix_per_cell=16, cell_per_block=2, transform_sqrt=False, vis=False):
	img = convert_color(img, 'RGB2'+ cspace)
	hist_features = color_hist(img, nbins=hist_bins)
	hog0 = get_hog_features(img[:,:,0], orient, pix_per_cell, cell_per_block, transform_sqrt=transform_sqrt)
	hog1 = get_hog_features(img[:,:,1], orient, pix_per_cell, cell_per_block, transform_sqrt=transform_sqrt)
	hog2 = get_hog_features(img[:,:,2], orient, pix_per_cell, cell_per_block, transform_sqrt=transform_sqrt)
	hogs = np.hstack((hog_feat1, hog_feat2, hog_feat3))
	return np.concatenate((hist_features, hogs))


def train_classifier():
	# Read in car and non-car images
	images = glob.glob('*.jpeg')
	cars = []
	notcars = []
	for image in images:
	    if 'image' in image or 'extra' in image:
	        notcars.append(image)
	    else:
	        cars.append(image)

	# TODO play with these values to see how your classifier
	# performs under different binning scenarios
	spatial = 32
	histbin = 32

	car_features = extract_features(cars, cspace='RGB', spatial_size=(spatial, spatial),
	                        hist_bins=histbin, hist_range=(0, 256))
	notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(spatial, spatial),
	                        hist_bins=histbin, hist_range=(0, 256))

	# Create an array stack of feature vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
	    scaled_X, y, test_size=0.2, random_state=rand_state)

	print('Using spatial binning of:',spatial,
	    'and', histbin,'histogram bins')
	print('Feature vector length:', len(X_train[0]))
	# Use a linear SVC 
	svc = LinearSVC()
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
	# Check the prediction time for a single sample
	t=time.time()
	n_predict = 10
	print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
	print('For these',n_predict, 'labels: ', y_test[0:n_predict])
	t2 = time.time()
	print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC') 