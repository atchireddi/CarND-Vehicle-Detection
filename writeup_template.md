 
## **Vehicle Detection Project**

#### **Objective:**
Identify and track vehicles of video stream. Its very important for an autonomous vehicle to identify/track
other vehicles/object of the environment. Having said that, this project is extensively focused on identifying cars
on the road. I will walk you through my pipeline for vehicle identification.
    
#### **Pipeline:**    

Below is an overview of vehicle detection pipeline. Description of each step follows.
Pipeline relies on  combination of HOG features, color spaces and spatial binning over a set of window sizes
for vehicle detection. A pre-trained SVM classifer was choosen for object classification task.
False positives were excluded through 6 frame ensamble.

```
def process_image(image):    
    # Try different window sizes
    windows = []
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[350, 500], 
                        xy_window=(96, 64), xy_overlap=(0.5, 0.5))
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[350, 500], 
                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[350, 500], 
                        xy_window=(128, 96), xy_overlap=(0.5, 0.5))
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 600], 
                        xy_window=(192, 128), xy_overlap=(0.5, 0.5))
    
    hot_windows = search_windows(image, windows, svc, X_scaler, 
                             color_space=color_space, 
                             spatial_size=spatial_size, 
                             hist_bins=hist_bins, 
                             orient=orient, 
                             pix_per_cell=pix_per_cell, 
                             cell_per_block=cell_per_block, 
                             hog_channel=hog_channel, 
                             spatial_feat=spatial_feat, 
                             hist_feat=hist_feat, 
                             hog_feat=hog_feat)                       

    # back-up hot windows over last 6 images
    if cnt == 1:
        for i in np.arange(6):
            buffer.append(hot_windows)
    
    # update buffer
    buffer.pop(0)
    buffer.append(hot_windows)
    
    window_img = mark_cars(image, [item for i in buffer for item in i])
    
    
    # Return image
    return window_img
    
```

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./output_images/boxes.jpg
[image4]: ./output_image/sampe_output.jpg
[image5]: ./output_images/ensamble.jpg
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4
 

### Histogram of Oriented Gradients (HOG):
HOG feature set, reduces the input dimension space for a classifier, while maintaining unique signature.
For a 64x64 3 channel image, HOG feature set reduces to 1769 with ` 8 pixel per cell`, `2 cells per block` 
and `9 gradient orientations`. I considered Histogram of Orientaed Gradients as one of feature set for 
vehicle/non-vehicle classification. ` hog()` funciton from `skimage.feature` was used to extract HOG features. 

Code for HOG feature extraction can be found as `get_hog_features()` function in `vehicleDetection.ipynb` file.

Below images illustrate extracted HOG features for `vehicle` and `non-vehicle` images.

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. **Classifier:**
I trained `linear SVM` classifier with combination of `HOG features` and color features from `YCrCb` color space. After
a bunch of trails, `YCrCb` color space characteristics appears to have linearly seperable features for cars & non-cars.

KITTI vehicle and non-vehicle datasets were train the classifier. since KITTI data is time series, for better optimization
data was shuffled and split into train and test sets. `extract_features()` function in `vehicleDetection.ipynb` file has code implementation for feature extraction. `StandarScaler()` from `sklearn.preprocessing` was used for feature normalization, to avoid feature biasing. 

### **Sliding Window Search:**

I performed a sliding window search for vehicle detection. since, the bottom half of the image is where most likely to find
a vechile, I limit my search region only to the bottom half of the image. To deal with far and near sight vehicles, I performed
multiple searches with different window sizes.

`process_image()` function in `vehicleDetection.ipynb` has implementation of different window sizes being considered.
below is the image showing search windows.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

Here's a [link to my video result](./output.mp4)


#### **False positives:**
Though above mentioned combination of HOG, histogram of color and spatially binned color features does really well on identifying
vehicles, I observe, there still false regions reported as vehicles. I adopted threshold technique to filter these false positivies.

I created a heatmap of all positive detections, I took an ensamble of positive detections from last 6 frames for heat map creation. Applied a threshold of 6, to identify true positive detections.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. assumed each blob corresponded to a vehicle.  A bounding boxe was created to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### **Discussion:**

My pipeline does pretty well detecting near vehicles, its sluggish on farsight vehicle detection. Though the classifier does
perform really well on validation set, it still has to identify all vechicles for each frame. I would think of additional training data to make my classifier more robust.

