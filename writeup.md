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

[img1.2.1]: ./output_images/step1_dataset_distribution.png "Dataset Distribution"
[img1.3]: ./output_images/step1_traffic_sign_images.png "Training Dataset Images of each Class/Label"
[img2.1.2]: ./output_images/step2_preprocessed_train_image.png "Train Dataset Image Pre Processed "
[img3.1.1]: ./output_images/step3_preprocessed_new_dataset.png "Test Dataset Images Pre Processed "
[img3.7.1]: ./output_images/1_munich_L3_FC1_297939.png "Prediction of image 1_munich.png"
[img3.7.2.1]: ./output_images/1_unknown_L3_FC1_297939.png "Prediction of image 1_unknown.png"
[img3.7.2.2]: ./output_images/2_unknown_L3_FC1_297939.png "Prediction of image 2_unknown.png"
[img3.7.2.3]: ./output_images/3_unknown_L3_FC1_297939.png "Prediction of image 3_unknown.png"
[img4.1]: ./output_images/1_munich_L3_FC1_297939_conv1.png "Convolutional Layer 1 visualization of L3_FC1_297939 model"
[img4.2]: ./output_images/1_munich_L3_FC1_297939_conv2.png "Convolutional Layer 2 visualization of L3_FC1_297939 model"
[img4.3]: ./output_images/1_munich_L3_FC1_297939_conv3.png "Convolutional Layer 3 visualization of L3_FC1_297939 model"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
## Writeup / README
&nbsp;

* Note: I used the given paper `Traffic Sign Recognition with Multi-Scale Convolutional Networks` as a start point of techniques used and reference.
  
>1 . Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rodriguesrenato/CarND-Traffic-Sign-Classifier-Project/blob/main/Traffic_Sign_Classifier.ipynb)

&nbsp;
## Data Set Summary & Exploration
&nbsp;

> 1 . Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and python libraries to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

The classes distribution along each dataset is shown below:

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

&nbsp;
> 2 . Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the classes distribution on train, validation and test dataset

![alt text][img1.2.1]

And here is a visualization of each traffic sign from the train dataset. To get images that aren't too bright or dark, the image sample mean value was checked. If this mean isn't between the specified threshold, it tests a new sample until get one that meets threshold criteria.

![alt text][img1.3]

&nbsp;
## Design and Test a Model Architecture
&nbsp;

> 1 . Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In section `2.1` of `Traffic_Sign_Classifier.ipynb`, I defined the function `pre_process_image()`, which receives an image as argument and returns the processed image. The following steps were taken:

1. Resize the image to 32 x 32 px. This makes the function useful in further steps to receive new images from web and assure consistence with defined model input shape

2. Convert the image to grayscale using opencv `cvtColor`. As described in the reference paper, grayscale input worked better than colored inputs, as well it will result in 1/3 on inputs to the model, less costly to train. When we observe the traffic signs, shown in section `1.3`, the shape and contours are more important than the color to classify the sign, there isn't any sign with the same contours, so a grayscale image would be valid to train. 
   
3. Normalize the image to a range of [-1,1] using opencv `normalize`. This step will make input with zero mean and equal variance, which is important to avoid numerical issues in the optimization process during training.

Here is an example of a traffic sign image before and after pre processing.

![alt text][img2.1.2]

I have mentioned in the notebook some future improvements for this section, such as increasing the training dataset and trying other image processing techniques, like histogram equalization to help increase contrast in dark and bright images. 

&nbsp;
> 2 . Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

For this project, two Models were designed based on LeNet-5:
- L2_FC1: **2 Convolutional Layers** (Convolution, Activation and Pooling) followed by a **1 Hidden Layer Classifier** (Fully Connected, Activation, Regularization Dropout and Fully Connected)
- L3_FC1: **3 Convolutional Layers** (Convolution, Activation and Pooling) followed by a **1 Hidden Layer Classifier** (Fully Connected, Activation, Regularization Dropout and Fully Connected)

The model that got the best results is `L3_FC1`, which is composed of the following layers:

| Stage | Layer             | Description               | Output Shape |
|----   |---                |---                        |---           |
| INPUT | Input             | Grayscale Image           | (32, 32, 1)  |
| L1    | Convolution 3x3   | 1x1 stride, valid padding | (30, 30, 32) |
| L1    | RELU              | Activation                |              |
| L1    | Max pooling       | 2x2 stride, 2x2 ksize     | (15, 15, 32) |
| L2    | Convolution 3x3   | 1x1 stride, valid padding | (13, 13, 64) |
| L2    | RELU              | Activation                |              |
| L2    | Max pooling       | 2x2 stride, 2x2 ksize     | (6, 6, 64)   |
| L3    | Convolution 3x3   | 1x1 stride, valid padding | (4, 4, 256)  |
| L3    | RELU              |                           |              |
| L3    | Max pooling       | 2x2 stride, 2x2 ksize     | (2, 2, 256)  |
| FC0   | Flatten           |                           | (1024)       |
| FC1   | Fully Connected   |                           | (100)        |
| FC1   | RELU              | Activation                |              |
| FC1   | Regulatization    | 0.7 Dropout               |              |
| FC1   | Fully Connected   | Logits                    | (43)         |



&nbsp;
> 3 . Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an optimizer called AdamOptimizer that minimizes the average cross entropy (loss). Cross entropy is calculated by applying softmax on logits and one_hot_encoding. 

A tensorflow session is started, variables initialized and training started. It begins by running the optimizer on batches of training sets, defined by BATCH_SIZE param, and when it has run all batches, calculates the validation accuracy. This operation is made the number of times defined by EPOCH param. At the end of EPOCH iterations, the test accuracy is calculated, along with calculating accuracy on the validation and whole training set.

The accuracy is calculated by the `evaluate()` function, which calculates the average of correct predictions on the given dataset. The dataset is divided and accuracy is calculated in batches of BATCH_SIZE param.

Finally, the results are saved with respective params in a file and the model is saved in the `trained_models` folder.

This whole procedure was designed in a function called `full_training_eval()`, which will be detailed in the next question. 

In `full_training_eval()`, these are the parameters that best performed: 
- Batch_size of 128
- Epoch of 100
- learning rate of 0.0005



&nbsp;
> 4 . Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

First of all, I have researched traffic sign recognition, read the reference paper to get insights of parameters values as starting points and model architectures.

I have implemented the classroom LeNet-5 model and got `0.89` test accuracy. After having a first version of training pipeline running, I tried the reference paper model architecture by joining the output of `L1` with the output of `L2`, but I couldn't get a high test accuracy. Then, I have designed the `L2_FC1` and `L3_FC1` with, respectively, 2 and 3 Convolutional Layers, followed by one fully connected hidden layer. I have tried to add more fully connected hidden layers and directy fully connect the last convolutional layer with the output, but the results were worse than with just one layer. Epochs iterations made great improvement to test results, so in the beginning I used a fixed value to compare other parameters and then I took the most promising ones to train with high epochs and lower training rates, because epochs increase the training time considerably.

The model started to get bigger/complex and to help control the overfitting detected by accuracy results, the regularization dropout method was added after the fully connected layer 1.

As the tests started taking longer to train, I decided to design a way to sequentially train all the desired parameters combinations. To achieve that, I coded the whole training + evaluation in a single function that receives a dictionary of parameters called `params`, which is responsible to choose the model and set all relevant variables.

On each training iteration, the results are saved on a file by `save_training_results()` and can be shown any time, ordered by test accuracy, in the console by running `print_training_results()`.

After many iterations, with multiple combinations of filter size, input/output depth, number of classifier hidden layer size, learning rate, epoch, etc, I got the following table:


| duration | valid_acc | test_acc |  arch  | l1_hw | l1_do | l2_hw | l2_do | l3_hw | l3_do | lfc1_d | reg_kp | EPOCH | BATCH_SIZE |  RATE  |
|----------|-----------|----------|--------|-------|-------|-------|-------|-------|-------|--------|--------|-------|------------|--------|
| 00:49:39 |   0.975   |  0.949   | L3_FC1 |  3.0  | 32.0  |  3.0  | 64.0  |  3.0  | 256.0 | 100.0  |  0.7   | 100.0 |   128.0    | 0.0005 |
| 00:37:01 |   0.969   |  0.943   | L3_FC1 |  3.0  | 32.0  |  3.0  | 64.0  |  3.0  | 256.0 | 100.0  |  0.7   | 75.0  |   128.0    | 0.0005 |
| 00:23:34 |   0.96    |   0.94   | L3_FC1 |  3.0  | 32.0  |  3.0  | 64.0  |  3.0  | 256.0 | 100.0  |  0.7   | 50.0  |   128.0    | 0.0005 |
| 00:17:41 |   0.955   |   0.94   | L2_FC1 |  7.0  | 16.0  |  5.0  | 64.0  |  3.0  | 256.0 | 100.0  |  0.7   | 50.0  |   128.0    | 0.0005 |
| 00:08:59 |   0.953   |  0.934   | L3_FC1 |  3.0  | 16.0  |  3.0  | 64.0  |  3.0  | 256.0 | 100.0  |  0.7   | 25.0  |   128.0    | 0.0005 |
| 00:18:06 |   0.948   |  0.933   | L3_FC1 |  3.0  | 16.0  |  3.0  | 64.0  |  3.0  | 256.0 | 100.0  |  0.7   | 50.0  |   128.0    | 0.0005 |
| 00:09:17 |   0.949   |  0.932   | L3_FC1 |  3.0  | 16.0  |  3.0  | 64.0  |  3.0  | 256.0 | 150.0  |  0.7   | 25.0  |   128.0    | 0.0005 |
| 00:07:39 |   0.954   |  0.929   | L3_FC1 |  3.0  | 16.0  |  3.0  | 64.0  |  3.0  | 128.0 | 100.0  |  0.7   | 25.0  |   128.0    | 0.0005 |
| 00:05:46 |   0.949   |  0.928   | L2_FC1 |  7.0  | 32.0  |  7.0  | 64.0  |  5.0  | 48.0  | 100.0  |  0.7   | 10.0  |   128.0    | 0.001  |
| 00:18:37 |   0.951   |  0.927   | L2_FC1 |  7.0  | 16.0  |  7.0  | 64.0  |  3.0  | 256.0 | 100.0  |  0.7   | 50.0  |   128.0    | 0.0005 |
| 00:05:46 |   0.938   |  0.924   | L3_FC1 |  3.0  | 16.0  |  3.0  | 32.0  |  3.0  | 128.0 | 100.0  |  0.7   | 25.0  |   128.0    | 0.0005 |
| 00:11:35 |   0.95    |  0.921   | L2_FC1 |  7.0  | 32.0  |  7.0  | 64.0  |  5.0  | 48.0  | 100.0  |  0.7   | 20.0  |   128.0    | 0.001  |
| 00:06:33 |   0.943   |  0.919   | L3_FC1 |  3.0  | 16.0  |  3.0  | 32.0  |  3.0  | 256.0 | 100.0  |  0.7   | 25.0  |   128.0    | 0.0005 |
| 00:09:08 |   0.936   |  0.918   | L3_FC1 |  3.0  | 16.0  |  3.0  | 64.0  |  3.0  | 256.0 |  50.0  |  0.7   | 25.0  |   128.0    | 0.0005 |
| 00:02:25 |   0.927   |  0.915   | L3_FC1 |  3.0  | 16.0  |  3.0  | 32.0  |  3.0  | 128.0 | 100.0  |  0.7   | 10.0  |   128.0    | 0.001  |
| 00:02:21 |   0.918   |  0.915   | L3_FC1 |  3.0  | 16.0  |  3.0  | 32.0  |  3.0  | 128.0 | 100.0  |  0.0   | 10.0  |   128.0    | 0.001  |
| 00:05:52 |   0.934   |  0.909   | L2_FC1 |  7.0  | 32.0  |  7.0  | 64.0  |  5.0  | 48.0  | 100.0  |  0.0   | 10.0  |   128.0    | 0.001  |

The model that achieve the best performance was `L3_FC1`, with 3 Convolutional layers (3x3 convolution filter size and 1x1 stride, Relu activation and 2x2 max pooling stride/ksize) and 1 Fully Connected hidden layer with size of 100, followed by dropout regularization of 0.7. 

This model was saved as `L3_FC1_297939` and its final results were:

* training set accuracy of `0.999` 
* validation set accuracy of `0.975` 
* test set accuracy of `0.949`

And these are parameters used on `L3_FC1_297939`

```python
params = {
    'arch': 2, 
    'EPOCH': 100, 
    'BATCH_SIZE': 128, 
    'RATE': 0.0005,
    'l1_hw': 3,
    'l1_di': 1,
    'l1_do': 32,
    'l1_st': 1,
    'l1_mp_ks': 2,
    'l1_mp_st': 2,
    'l2_hw': 3,
    'l2_do': 64,
    'l2_st': 1,
    'l2_mp_ks': 2,
    'l2_mp_st': 2,
    'l2b_mp_ks': 4,
    'l2b_mp_st': 4,
    'l3_hw': 3,
    'l3_do': 256,
    'l3_st': 1,
    'l3_mp_ks': 2,
    'l3_mp_st': 2,
    'lfc1_d': 100,
    'lfc2_d': 101,
    'reg_kp': 0.7
    }
```

Even though these results show an overfitting by the test accuracy is lower than the validation accuracy, the test accuracy got was over 93% and performed well in following sections.

I could only get high accuracy when increasing the network complex by adding more layers and increasing the output depth of convolutional layers. Regularization helped a lot to get high accuracies, but not enough to make accuracies closer.

I think if I get a more uniform and bigger training dataset on each class, it would help getting better accuracies, as I noticed that lower complex networks haven't performed well.

&nbsp;
### Test a Model on New Images
&nbsp;

> 1 . Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

To get new German traffic sign images, Google street views was opened on roads near Maximilianeum in Munich, Germany, to take screenshots of traffic signs and save them in test_images folder. The filenames pattern used is: [sign class]_[place name].png. The images have different sizes because it was a manual screenshot direct from google maps in the browser, so each image is cropped (crop vertices manually defined in code) and pre processed with `pre_process_image()` function used previously.

These are the images in the new test dataset and the respective pre processed version:

![alt text][img3.1.1]

The `[29] Bicycle crossing` image might be difficult to classify if we check the pre processed generated image. The bicycle lines are too thin at 32x32 size, losing part of the contours and pixels. All other images with thin lines and areas would be difficult to process, as well the ones with similar shape, like speed limits. Although, all images were successfully recognized by the model.

&nbsp;
> 2 . Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy of these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Test results with model: L3_FC1_297939

| Test Img #  |               Sign Name               | Sign Class/Label | 1st Softmax Prediction | 2st Softmax Prediction | 3st Softmax Prediction | 4st Softmax Prediction | 5st Softmax Prediction |
|---|---|---|---|---|---|---|---|
|   0   |         Speed limit (30km/h)          |  1   | 1  [1.0e+00] | 2  [1.3e-23] | 0  [2.0e-35] | 3  [0.0e+00] | 4  [0.0e+00] |
|   1   |         Speed limit (50km/h)          |  2   | 2  [1.0e+00] | 1  [1.1e-17] | 40 [5.1e-28] | 37 [5.3e-31] | 21 [7.7e-33] |
|   2   | Right-of-way at the next intersection |  11  | 11 [1.0e+00] | 30 [1.7e-11] | 21 [4.9e-17] | 40 [2.6e-21] | 42 [1.1e-22] |
|   3   |             Priority road             |  12  | 12 [1.0e+00] | 40 [7.4e-17] | 42 [2.2e-19] | 38 [5.1e-20] | 36 [4.5e-21] |
|   4   |                 Yield                 |  13  | 13 [1.0e+00] | 35 [1.5e-20] | 9  [4.7e-21] | 41 [1.8e-22] | 1  [1.6e-25] |
|   5   |              No vehicles              |  15  | 15 [1.0e+00] | 1  [3.5e-09] | 2  [4.0e-11] | 26 [7.7e-12] | 38 [3.0e-12] |
|   6   |               No entry                |  17  | 17 [1.0e+00] | 23 [3.6e-10] | 10 [1.7e-13] | 37 [4.6e-15] | 20 [5.4e-16] |
|   7   |            Traffic signals            |  26  | 26 [1.0e+00] | 29 [1.2e-08] | 24 [6.0e-09] | 8  [7.2e-11] | 25 [4.2e-12] |
|   8   |              Pedestrians              |  27  | 27 [1.0e+00] | 11 [3.6e-09] | 24 [2.5e-09] | 28 [2.5e-13] | 21 [2.7e-14] |
|   9   |           Bicycles crossing           |  29  | 29 [1.0e+00] | 23 [2.3e-09] | 24 [1.1e-09] | 28 [2.7e-10] | 30 [6.3e-14] |
|  10   |           Turn right ahead            |  33  | 33 [1.0e+00] | 36 [4.4e-15] | 35 [1.7e-18] | 34 [1.4e-19] | 14 [9.2e-21] |
|  11   |              Ahead only               |  35  | 35 [1.0e+00] | 9  [1.4e-12] | 33 [2.5e-14] | 3  [7.9e-15] | 34 [1.6e-15] |
|  12   |         Go straight or right          |  36  | 36 [1.0e+00] | 42 [5.6e-11] | 38 [3.8e-11] | 32 [1.1e-11] | 6  [6.0e-12] |
|  13   |              Keep right               |  38  | 38 [1.0e+00] | 10 [2.2e-32] | 31 [5.1e-36] | 0  [0.0e+00] | 1  [0.0e+00] |
|  14   |              Keep right               |  38  | 38 [1.0e+00] | 0  [0.0e+00] | 1  [0.0e+00] | 2  [0.0e+00] | 3  [0.0e+00] |
|  15   |               Keep left               |  39  | 39 [1.0e+00] | 33 [1.4e-16] | 13 [9.4e-22] | 14 [6.2e-22] | 40 [2.7e-22] |


The model was able to correctly guess all traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.9%. It used images with good lighting conditions, directly from google maps. Even though there were any distortions in shape, the model performed well.

&nbsp;
> 3 . Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the section `3.` of the Ipython notebook. From section `3.1` to `3.6`, it was implemented the pre processing, prediction pipeline, performance analyzation and softmax output for all new test images. In section `3.7` there is a implementation to run prediction on a single image 

The 5 softmax results for each image are shown in the previous question.

This is a visualization of the result of a recognition in section `3.7` on the `1_munich.png` image:

![][img3.7.1]

The first softmax probability is set 100%, probably rounded up as the following softmax probabilities are too small, because its order of magnitude ranges from -29 to -39.

It is the same for the other images, the second softmax probabilities order of magnitude range from -08 to -32, which make the first softmax rounded to 100%.

As the model recognizes all test images, I decided to test the model with untrained traffic signs:

![][img3.7.2.1]
1. [1] Speed limit (30km/h)
2. [6] End of speed limit (80km/h)
3. [0] Speed limit (20km/h)
4. [5] Speed limit (80km/h)
5. [4] Speed limit (70km/h)

The results show that the input image tends to be a traffic sign that has a circular shape and a red border. The top 5 softmax probabilities have closer order of magnitude than the one tested with a known traffic sign by the model, showing that the probabilities are more distributed over the labels due to the prediction uncertainty.

![][img3.7.2.2]
1. [40] Roundabout mandatory 
2. [06] End of speed limit (80km/h)
3. [01] Speed limit (30km/h)
4. [04] Speed limit (70km/h)
5. [07] Speed limit (100km/h)

In addition to the previous prediction, the probabilities are distributed between rounded traffic signs that have a red border.

![][img3.7.2.3]

1. [14] Stop
2. [09] No passing
3. [36] Go straight or right
4. [02] Speed limit (50km/h)
5. [35] Ahead only

It is interesting that the results got 100% for Stop Sign, but as we can see the other 4 softmax have similar orders of magnitude around -8. The result might be explained by the fact that the image has mostly an uniform color with a thin white border and some straight and curved shapes on the center. The color isn't considered  because the images were grayscaled to feed the model, when blue and red are grayscaled, it got a similar value.

&nbsp;
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
&nbsp;

> 1 . Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I used the `1_munich.png` image to show the neural network layers.

This is the first convolutional layer output:

![][img4.1]

The circular shape/contour of the traffic sign and the numbers are highlighted. It is visible that the contour of the red border is shown in many Feature Maps (seems it shows the internal and external contours of the red border).

Second convolutional layer output:

![][img4.2]

it is a bit harder to find patterns, the image size is too small. The FeatureMap 25 and 29 seems to refers to the circular border pattern, but from this point it is difficult to infer any pattern. The image bellow is from the thrid convolutional layer output:

![][img4.3]