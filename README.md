##Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

The goal of this project was:

- Identify vehicles in images

- Track vehicles across frames in a video stream

The main steps followed were:

- Feature Extraction on labeled vehicle images

    - Histogram of Oriented Gradients (HOG)

    - Histogram of colors

    - Spatial Bining

- Feature Normalization your features and

- randomize a selection for training and testing

- Train a classifier using different methods

- Implement a sliding-window technique and use your trained classifier to search for vehicles in images

- Run your pipeline on a video stream

- Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles

- Estimate a bounding box for vehicles detected


[//]: # (Image References)

[image1]: ./examples/data_set.png "Training set"
[image2]: ./examples/hog.jpg "HOG Representation"
[image3]: ./examples/output_bboxes.png "Feature Detection"
[image4]: ./examples/Hog16.png "Feature Detection"
[image5]: ./examples/test_images_feature.png "Feature Detection on Data Set"
[image6]: ./examples/bboxes_and_heat.png "Feature Detection on Data Set"


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images. Here are some examples of vehicle and non vehicle images.

![alt text][image1]

The following three features were used:

#### Histogram of Colors:
- Color histogram of each color channel is calculated and concatenate together. Histogram os divided into 16 bins. Adding the three bins, this returns a total of 96 features.

#### Spatial Binning
- Spatial binning is done after resizing the image to 16x16. It flattens the image array. So for a 16x16 image with 3 channels, we get 768 features.

#### Histogram of Gradients(HOG)
- The image is converted to the YUV colorspace. Different color spaces were tried like RGB, HSV & HSL, and YCrCb
- HOG for were generated for each color channel with 9 orientations, 8 pixels per cell, and 2 cells per block. For a 64x64 image, this returns 1764 features per channel or an array of shape (7, 7, 2, 2, 9) before flattening. That's a total of 5292 features.

Hence there is a total of 6108 features per image.

This is an example of feature detection on one image
 ![alt text][image3]

I explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Y` color channel and HOG parameters of `orientations=9`, `pixels_per_cell=(9, 9)` and `cells_per_block=(2, 2)` compared with the original images.

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

Function extract_features was defuned that took an array of 64x64x3 images and returned a set of features. Paramteres tuned for this function were:
- Image size for spatial bining (spatial_size)
- Number of histogram bins (hist_bins)
- Number of orientations (orient)
- Number of pixels per cell (pix_per_cell)
- Number of cell per block (cell_per_block)
- Number of hog channels (hog_channel)
- Color space (color_space)

Here I will show how the HOG Parameters effect the claasification efficiency. I will use the following configurations where I changed the color space and calculated the classification efficiency using Linear SVC method


|    Configuration    | Colorspace | Orientations | Pixels Per Cell | Cells Per Block | HOG Channel |SVC Accuracy |
| :-----------------: | :--------: | :----------: | :-------------: | :-------------: | :---------: |:---------:  |    
| 1                   | RGB        | 9            | 8               | 2               | 0           |  93.44      |
| 2                   | LUV        | 9            | 8               | 2               | 0           |  95.07      |
| 3                   | HLS        | 9            | 8               | 2               | 0           |  90.68      |
| 4                   | YUV        | 9            | 8               | 2               | 0           |  96.38      |
| 5                   | YCrCb      | 9            | 8               | 2               | 0           |  94.85      |

As can be seen color space 'YUV' gave the best results and was hence chosen. Similar steps were followed by varying other paramters and seeing their effect on cllasification accuracy.
The final parameter settings used were spatial_size = (16,16), hist_bins = 16, orient = 9, pix_per_cell = 8 , cell_per_block = 2 hog_channel = 'ALL', and  color_space = 'YUV'

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).


The model is trained using different classification methods:
- Linear SVC
- Logistic Regression
- Multi layer perceptron (MLP)
- Decision Tree
- Random Forests

The MLP classifier with an accuracy score of 0.993 was chosen for this project

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

A single function find_cars is defined that can extract features using hog sub-sampling and make predictions
The method combines HOG feature extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the HOG features (and other features) are extracted for the entire image (or a selected portion of it) and then these full-image features are then fed to the classifier. The method performs the classifier prediction for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.

In the find_cars function each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows.

I explored several configurations of window sizes and positions, with various overlaps in the X and Y directions using scale factor small (1x), medium (1.5x, 2x), and large (3x) windows. Smaller (0.5) scales were explored but found to return too many false positives. The final implementation considers 1.5x scale factor. To get better and more robust results multiple-scaled search windows should be used.  

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The final set of parameters I used were HOG features on YCrCb channels, 32 histogram bins on the YCrCb image, 32 histogram bins on the HLS image and (16,16) spatial bins. I started out with 16 histogram bins, but had trouble detecting white cars on the road, especially in bright areas. Increasing the number of histogram bins to 32 seemed to train the classifier better. I also added histogram bins from the HLS channels to fix this issue. I searched the image using two scales. However, in order to optimize the performance of the classifier, I restrict the search area of the second scale to Y-values of 350-474 (124 pixels close to the horizon). The scale for this second search is smaller, due to the fact that cars close to the horizon appear small. Here are some examples of the search algorithm some text images: ![alt text][image5]


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)


The add_heat function increments the pixel value (referred to as "heat") of an all-black image the size of the original image at the location of each detection rectangle. Areas encompassed by more overlapping rectangles are assigned higher levels of heat

apply_threshold function is applied to the heatmap, setting all pixels that don't exceed the threshold to zero

draw_labeled_bboxes function gives the final detection area based on the extremities of each identified label

Here's a [link to my video result](./project_video_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected (code cell 25-26).

At this point, I processed the test video and realized that while the detections on the vehicles are pretty accurate, the bounding boxes jump around and are not stable. In order to smooth the detection over time, I average the heat map over the past five frames, and this resulted in much smoother detections.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here are things that are likely to fail in the pipeline and some possible fixes:
1. When two cars are close to each other they get detected as one box, as the label function detects them as one blob.
2. There are still some issues detecting white cars on light road surfaces. This could potentially be an issue with the classifier - Data augmentation or using a better classifier would improve this.

In order to continue with the projects, the following are good future steps to take:
1. I would start out by equalizing the brightness and contrast on the training and test set before training the classifier.
2. I would also use data augmentation to improve the detection of white cars on light roads.

The algorithm seems quite robust but it still falsely identifies some non-vehicle objects such as trees and traffic signs. In order to improve performance we need to probbably increase the size training dataset. Plus since the parameters were manually fine-tuned it is possible that the algorithm is over-fitted on the specific video and hence change in conditions like different lighting conditions, shadows, urban/highway/rural driving might deteriorate the performance.

Moreover, vehicles with a high-speed difference to camera won’t be tracked correctly based on the fact that detections are averaged over several frames. So the detection boxes would lag behind the visible vehicle.

Other kinds of motor vehicles like trucks, motorbikes, cyclists (and pedestrians) would not be detected (as they are not in the training data)

I think using computer vision is an interesting approach but convolutional neural network approach may show better and more robust results.
