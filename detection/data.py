#!/bin/python

import numpy as np 
import cv2
import pickle
import os
import os.path
import zipfile
from urllib.request import urlretrieve
import pickle
from detection.file import full_path
import glob
import matplotlib.image as mpimg

"""
A list of urls to download the data from. Helpful if working on multiple machines.
"""
small_non_vehicle_url = "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip"
small_vehicle_url = "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip"
full_non_vehicle_url = "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip"
full_vehicle_url = "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip"

"""
A list of file urls to find the data in
"""
file_urls = {
	"small": {
		"vehicle": {
			"url": small_vehicle_url,
			"folder": "vehicles_smallset",
		}, "non-vehicle": {
			"url": small_non_vehicle_url,
			"folder": "non-vehicles_smallset"
		}
	},
	"full": {
		"vehicle": {
			"url": full_vehicle_url,
			"folder": "vehicles"
		}, "non-vehicle": {
			"url": full_non_vehicle_url,
			"folder": "non-vehicles"
		}
	}
}

def load_data(size='small'):
	"""
	Loads one of two data sets, `. Function works no matter if the data exists or not.
	First downloads the data if it isnt there, then unzips the data if it hasn't been unzipped. 
	Then collects and returns it in a consistent format.

	:param size: String, one of two datasets, `small` or `full`
	:returns: Tuple, tuple of list of vehicle and list of non-vehicle images.
	"""
	folder_path = full_path("data/"+size)
	if os.path.isdir(folder_path) == False:
		print("Unable to find", size, "creating a folder for it.")
		os.mkdir(folder_path)

	download_zip(folder_path, size, "vehicle")
	download_zip(folder_path, size, "non-vehicle")
	unzip_file(folder_path, size, "vehicle")
	unzip_file(folder_path, size, "non-vehicle")

	vehicle_images = load_images(folder_path + "/" + file_urls[size]["vehicle"]["folder"])
	non_vehicle_images = load_images(folder_path + "/" + file_urls[size]["non-vehicle"]["folder"])

	print("Found", len(vehicle_images), "vehicle images and", len(non_vehicle_images), "non-vehicle images in data folder.")

	return (
		# process_images("vehicle", vehicle_images), 
		# process_images("non-vehicle", non_vehicle_images)
		vehicle_images, non_vehicle_images
	)

def load_images(path):
	"""
	Loads images from folders inside a file path.
	:param path: path to image folders
	"""
	result = []
	for folder in os.listdir(path + "/"):
		for file in glob.glob(path + "/" + folder + "/*.jpeg"):
			result.append(mpimg.imread(file))
		for file in glob.glob(path + "/" + folder + "/*.png"):
			img = cv2.imread(file)
			b,g,r = cv2.split(img)       # get b,g,r
			rgb_img = cv2.merge([r,g,b]) 
			result.append(rgb_img)
	return result

def process_images(name, images):
	"""
	Flips images horizontally to increase image number.
	:param name: name of image set
	:param images: list of images to filp
	"""
	print("Processing", name, "images.")
	for i in range(len(images)):
		images.append(np.fliplr(images[i]))
	return images

def download_zip(folder_path, size, type):
	"""
	download zip file to path if it doesnt exists
	:param folder_path: path to look for zip file
	:param size: dataset, `small` or `full`, used in dictionary
	:param type: type of image, used in dictionary above.
	"""
	path = "{}/{}.zip".format(folder_path, type)
	download_file(file_urls[size][type]["url"], path)

def unzip_file(folder_path, size, type):
	"""
	unzip file to path if it hasn't been unzipped
	:param folder_path: path to look for unzip folders
	:param size: dataset, `small` or `full`, used in dictionary
	:param type: type of image, used in dictionary above.
	"""
	path = "{}/{}".format(folder_path, type)
	path_result = "{}/{}".format(folder_path, file_urls[size][type]["folder"])
	if os.path.isdir(path_result) == False:
		print("Unable to find " + path_result + ". Unzipping now...")
		unzip_data(path + ".zip", folder_path)
	else:
		print(path_result, "already unzipped.")

def download_file(url, file):
	"""
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
	if os.path.isfile(file) == False:
		print("Unable to find " + file + ". Downloading now...")
		urlretrieve(url, file)
		print('Download Finished!')
	else:
		print(file + " already downloaded.")

def unzip_data(zip_file_name, location):
	"""
    unzip file 
    :param zip_file_name: name of zip file
    :param location: path to unzip location
    """
	with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
		print("Extracting zipfile " + zip_file_name + "...")
		zip_ref.extractall(location)

def open_pickle_file(file_name):
	"""
    open a pickled file
    :param file_name: name of file
    """
	print("Unpickling file " + file_name + ".")
	full_file_name = full_path(file_name)
	with open(full_file_name, mode='rb') as f:
		return pickle.load(f)

def save_pickle_file(file, data):
	"""
    save an object as a pickled file
    :param file: name of file
    :param data: python object to save
    """
	abs_file = full_path(file)
	pickle.dump(data, open(abs_file, "wb" ) )
