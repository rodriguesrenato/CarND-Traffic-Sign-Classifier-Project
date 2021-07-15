# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[img1.2.1]: ./output_images/step1_dataset_distribuition.png "Dataset Distribuition"
[img1.3]: ./output_images/step1_traffic_sign_images.png "Training Dataset Images of each Class/Label"
[img2.1.2]: ./output_images/step2_preprocessed_train_image.png "Train Dataset Image Pre Processed "
[img3.1.1]: ./output_images/step3_preprocessed_new_dataset.png "Test Dataset Images Pre Processed "
[img3.8.1]: ./output_images/1_munich_L3_FC1_297939.png "Prediction of image 1_munich.png"
[img3.8.2.1]: ./output_images/1_unknown_L3_FC1_297939.png "Prediction of image 1_unknown.png"
[img3.8.2.2]: ./output_images/1_unknown_L3_FC1_297939.png "Prediction of image 2_unknown.png"
[img3.8.2.3]: ./output_images/1_unknown_L3_FC1_297939.png "Prediction of image 3_unknown.png"
[img4.1]: ./output_images/1_munich_L3_FC1_297939_conv1.png "Convolutional Layer 1 visualization of L3_FC1_297939 model"
[img4.2]: ./output_images/1_munich_L3_FC1_297939_conv1.png "Convolutional Layer 2 visualization of L3_FC1_297939 model"
[img4.3]: ./output_images/1_munich_L3_FC1_297939_conv1.png "Convolutional Layer 3 visualization of L3_FC1_297939 model"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
## Writeup / README

* Note: I used the given paper `Traffic Sign Recognition with Multi-Scale Convolutional Networks` as a start point of technics used and reference.
  
>1 . Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

## Data Set Summary & Exploration

> 1 . Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and python libraries to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

The classes distribuition along each dataset is shown below:

| Classes   | Traffic Sign Name | Training Images | Validation Images | Test Images |
|-----------|----------------------------------------------------|------|-----|-----|
|         0 | Speed limit (20km/h)                               |  180 |  30 |  60 |
|         1 | Speed limit (30km/h)                               | 1980 | 240 | 720 |
|         2 | Speed limit (50km/h)                               | 2010 | 240 | 750 |
|         3 | Speed limit (60km/h)                               | 1260 | 150 | 450 |
|         4 | Speed limit (70km/h)                               | 1770 | 210 | 660 |
|         5 | Speed limit (80km/h)                               | 1650 | 210 | 630 |
|         6 | End of speed limit (80km/h)                        |  360 |  60 | 150 |
|         7 | Speed limit (100km/h)                              | 1290 | 150 | 450 |
|         8 | Speed limit (120km/h)                              | 1260 | 150 | 450 |
|         9 | No passing                                         | 1320 | 150 | 480 |
|        10 | No passing for vehicles over 3.5 metric tons       | 1800 | 210 | 660 |
|        11 | Right-of-way at the next intersection              | 1170 | 150 | 420 |
|        12 | Priority road                                      | 1890 | 210 | 690 |
|        13 | Yield                                              | 1920 | 240 | 720 |
|        14 | Stop                                               |  690 |  90 | 270 |
|        15 | No vehicles                                        |  540 |  90 | 210 |
|        16 | Vehicles over 3.5 metric tons prohibited           |  360 |  60 | 150 |
|        17 | No entry                                           |  990 | 120 | 360 |
|        18 | General caution                                    | 1080 | 120 | 390 |
|        19 | Dangerous curve to the left                        |  180 |  30 |  60 |
|        20 | Dangerous curve to the right                       |  300 |  60 |  90 |
|        21 | Double curve                                       |  270 |  60 |  90 |
|        22 | Bumpy road                                         |  330 |  60 | 120 |
|        23 | Slippery road                                      |  450 |  60 | 150 |
|        24 | Road narrows on the right                          |  240 |  30 |  90 |
|        25 | Road work                                          | 1350 | 150 | 480 |
|        26 | Traffic signals                                    |  540 |  60 | 180 |
|        27 | Pedestrians                                        |  210 |  30 |  60 |
|        28 | Children crossing                                  |  480 |  60 | 150 |
|        29 | Bicycles crossing                                  |  240 |  30 |  90 |
|        30 | Beware of ice/snow                                 |  390 |  60 | 150 |
|        31 | Wild animals crossing                              |  690 |  90 | 270 |
|        32 | End of all speed and passing limits                |  210 |  30 |  60 |
|        33 | Turn right ahead                                   |  599 |  90 | 210 |
|        34 | Turn left ahead                                    |  360 |  60 | 120 |
|        35 | Ahead only                                         | 1080 | 120 | 390 |
|        36 | Go straight or right                               |  330 |  60 | 120 |
|        37 | Go straight or left                                |  180 |  30 |  60 |
|        38 | Keep right                                         | 1860 | 210 | 690 |
|        39 | Keep left                                          |  270 |  30 |  90 |
|        40 | Roundabout mandatory                               |  300 |  60 |  90 |
|        41 | End of no passing                                  |  210 |  30 |  60 |
|        42 | End of no passing by vehicles over 3.5 metric tons |  210 |  30 |  90 |

> 2 . Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the classes distribuition on train, validation and test dataset

![alt text][img1.2.1]

And here is a visualization of each traffic sign from train dataset. To get images that arent too bright or dark, the image sample mean value was checked, if it isnt between a estipulated threshold, it tests a new sample until get one that meets threshold criteria.

![alt text][img1.3]

## Design and Test a Model Architecture

> 1 . Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In section `2.1` of `Traffic_Sign_Classifier.ipynb`, I defined the function `pre_process_image()`, which receives an image as argument and return the processed image. The following steps were taken:

1. Resize the the image to 32 x 32 px. This makes the function useful in further steps to receive new images from web and assure consistence with defined model input shape

2. Convert the image to grayscale using opencv `cvtColor`. As described in the referece paper, grayscale input worked better than colored inputs, as well it will result in 1/3 on inputs to the model, less costly to train. When we observe the traffic signs, shown in section `1.3`, the shape and contours are more important than the color to classify the sign, there isn't any sign with same contours, so grayscale image would be valid to train. 
   
3. Normalize the image to a range of [-1,1] using opencv `normalize`. This step is importante to make 

Here is an example of a traffic sign image before and after pre processing.

![alt text][img2.1.2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


