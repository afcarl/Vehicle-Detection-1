## Vehicle Detection

In this project I will demonstrate how to locate vehicles in an image/video using traditional imageprocessing techniques such as [HOG features](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients).

[//]: # (Image References)

[image1]: ./output_images/car_image_pre_3d.png "Car Pre 3D Image"
[image2]: ./output_images/car_image_3d.png "Car 3D Image"
[image3]: ./output_images/non_car_image_pre_3d.png "Non Car Pre 3D Image"
[image4]: ./output_images/non_car_image_3d.png "Non Car 3D Image"
[image5]: ./output_images/car_image_pre_hog.png "Car Pre HOG Image"
[image6]: ./output_images/car_image_hog.png "Car HOG Image"
[image7]: ./output_images/non_car_image_pre_hog.png "Non Car Pre HOG Image"
[image8]: ./output_images/non_car_image_hog.png "Non Car HOG Image"
[image9]: ./output_images/car_image_heatmap.png "Heatmap Image"
[image10]: ./output_images/sliding_window.png "Sliding window"

## Step 1: Extract Features
I extracted three types of features in each image. All three techniques can be found in `detection/shared_functions.py`. The first, `bin_spatial`, is simply downsizing the image and then unrolling the image into a single array for each color channel. I didn't find this feature to be extremely helpful. The next, `color_hist`, is a histogram of colors per channel. Cars and really any human created objects seemed to have very different saturation and brightness values than those that occur in nature. See example below. 

Car Image
![alt image][image1]

3D pixel representation
![alt image][image2]

Non car image
![alt image][image3]

3D pixel representation
![alt image][image4]

I felt that they were significantly different enough to provide some distinction to the classifier. The last feature, `get_hog_features`, creates a histogram of orients to find edges in an image. Cars have distinct orientation lines in an image. This is a really powerful way to detect shapes in an image. 

Car Image
![alt image][image5]

HOG representation
![alt image][image6]

Non car image
![alt image][image7]

HOG representation
![alt image][image8]

Through a process of trial and error I decided to use the following params with the hog function: 
- YCrCb color space
- 8 orientations
- 16 pixels per
- 2 cells per block
- Apply power law compression to normalize the image before processing

I settle on these params based on a mixture of speed and accuracy. The biggest speed draw back was decreasing the number of pixels per cell to 8. This quadruples the number of features in a single hog image although I did see some accuracy gains in the classifier. Another speed/accuracy trade off is the number of cells per block. I settled on two although 1 or 3 could also have worked. I applied the power law compression to reduce the effects of shadows. The YCrCb color space performed significantly better than the RGB and HSV color spaces. The HSV color space removes a lot of the shadow edges in an image which is exactly what we want to look for. The actual feature extraction happens in the file `detection/extract_features` function `extract_features`.

## Step 2: Train Classifier
After extracting the features, I trained a simple classifier. This happens in the file `detection/extract_features` function `train_classifier`. The data was ~9K 64x64 images of vehicles and ~9K images of non-vehicles at varying quality levels (i.e. cars at different distances). Before the classifier is fit, the data is scaled using a `StandardScaler` and split into test and train data in file `detection/extract_features` function `test_train_data`. I again had to balance speed vs. accuracy. I first started out with an `AdaBoost` classifier, which is a ensemble classifier, is normal very robust with large sets of data. In this project, I had good accuracy and was able to prevent most over-fitting with the AdaBoost classifier. Unfortunately, AdaBoost is really slow at processing a single set of features when applying the classifier to the video. In this specific project speed at processing time wasn't an actual consideration but In the real world, a classifier should work in under 100ms and the Adaboost classifier took longer than a second per image. I next tried a `LinearSVC` classifier. Using the LinearSVC classifier I found that I was able to get better accuracy on the train set but the classifier was more over-fit than the Adaboost classifier. This downside was contrasted with a much faster prediction time, closer to 5ms. 

## Step 3: Apply classifier to road image  using sliding window
Once the classifier was trained, I had to apply it to an actual image (in our case 720x1280). I used a sliding window technique to search the image at different scales. This is done in file `detection/find_cars.py` function `find_cars`. 

![alt image][image10]

I used the find_cars function similar to the once in the Udacity videos. It applies the hog function to the entire images once because this is an expensive operation and then iterates over the defined pixel indices and uses the trained classifier to predict if a car is present in a 64x64 snap shot of the image. If the classifier predicts a car, then the points are saved and returned at the end of the function. This approach is faster than having to calculate the hog features at every step because there is overlap when doing the hog sampling. In my case I used a cell overlap of 1 which means I only shift over a single cell (16 pixels in my case) when searching the image. I used a small overlap because I observed that I was "missing" cars that were far away with a larger value. 

When searching the image is applied a scaling factor to grow/shrink the image to look for cars at different distances in the image. I found that using a scaling factor smaller than 1 slowed down the processing time for an image significantly. My final implementation uses scaling factors of 1 and 2. The downside is a sometimes lose the cars when they get a certain distance away. 

![alt image][image9]

## Step 3.1: Classifier Improvements
Overall I feel like I am getting too many false positives with my classifier and not enough true positives. Tweaking the threshold number became a very difficult process but I think it shouldn't be so sensitive because it means in other situations my predictions will fail. The training data was a lot of images of the same cars which I think hurt my training performance overall. It would have been better to have less images of the same car and more images of different types of cars with different colors and orientations.

In looking at the false positives, I think my classifier is mostly looking for bright/high saturation objects with lots of clean lines. This type of classification is helpful for identifying cars but could also apply to other sorts of human created objects with we may or may not want to avoid, consider in other ways (i.e. cones). A deeper dive into negative results would be needed to come up with a more complete understanding. Another solution could be to train a neural network rather than rely on a mixture of hog features and a linear classifier. 

## Step 4: Apply classifier to video
Applying the classifier and sliding window technique to the video is similar to applying to to a single image except we are able to leverage previous frame detections when drawing the bounding boxes. I used the last 8 frames when collecting and drawing the bounding boxes. This made the solution more robust as it is able to remove false positives from a single frame. This is done in `detection/road.py`. 

[![Project Video Link](https://j.gifs.com/oYlLxj.gif)](https://youtu.be/sp2r754g_Xk)

## Reflections
Overall this was again a difficult project because there were so many params to tune. Selecting a single param in isolation is normally not very difficult but obviously all the params are intertwined with each other so any one decision effects all the other ones. In some ways I feel like I am the Stochastic Gradient Decent optimizer slowly working my around the feature space looking for an optimized answer. 

Currently my solution works for the sunny road with two cars but I think it would have a hard time in traffic and in other lighting conditions. This is definitely something to test and optimize in the future. 

I think the biggest disadvantage to my solution is that its slow. Specifically it take stop long to create the hog feature space and search it for 64x64 images that the classifier predicts to be cars. In practice I think the whole process can take no longer than 10ms. Maybe even less. I have read that there are some faster neural network solutions which I want to explore next. 
