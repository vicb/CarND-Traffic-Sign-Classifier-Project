# Traffic Sign Recognition** 

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

[image1]: ./writeup/features_by_class.png "Features by class"
[image3]: ./writeup/synthetic_image.png "Synthetic feature"
[image4]: ./80.png "80"
[image5]: ./ahead.png "ahead"
[image6]: ./children.png "children"
[image7]: ./stop.png "stop"
[image8]: ./right.png "right"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
It is a bar chart showing the number of features per class

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


I decided not to convert the images to grayscale because it seems to me that colors could help identify the signs.

However it seems like converting to grayscale has proved to improve the accuracy as reported in [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It would be worth trying with the current model.

I normalized the image data because it's generally a good idea both to help the gradient descent converge faster
and to have smaller parameters to avoid numerical errors.

I decided to generate additional data because my first classifier was not good enough with the initial training set.
Also some of the classes had very little featues compared to the rest of them.

To add more data to the the data set, I created new images from the initial training set by applying a random rotation
followed by a random translation.

Here is an example of an original image and an augmented image:

![alt text][image3]

The augmented data set has 3 times the number of images present in the original data set.
It helped improve the accuracy by 0.5%. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| tf.nn.lrn             | local resp. normalization                     |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16     				|
| tf.nn.lrn             | local resp. normalization                     |
| Fully connected		| layers size: 400 -> 120 -> 84 -> 43    		|
| Softmax				|            									|
 
I first used the LeNet architecture as seen in the course.
The [result](Traffic_Sign_Classifier-1-LeNet-91.1.ipynb) was not good enough then I modify the model to use:
- local response normalization,
- dropout,
- L2 regularization.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
- an Adam optimizer,
- a constant learning rate of 0.001,
- a BETA of 0.0001,
- a dropout with a keep probability of 0.5,
- a batch size of 128,
- 15 epochs

Adding regularization (L2, dropout and lrn) has proven to be very efficient and the accuracy jumped from 91.1 to 93.8%
on the validation set.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.6%  
* test set accuracy of 93.8%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I first used the LeNet architecture as seen in the course.

* What were some problems with the initial architecture?

The accuracy was too low.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

To reduce overfitting, I have created more images (synthetic images) and added regul (L2, dropout and lrn).

* Which parameters were tuned? How were they adjusted and why?

I mostly adjusted the number of epochs (increasing from 10 to 15) after having added regul.
This was possible because regularization preven over-fitting.
Changing the other parameters (learning rate, BETA) had less impact on the accuracy.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolution layers have mutliple benefit. 
First they detect shapes and colors in the input images and they reduce the dimension of the input for the next layers.
To some extent, convolution layers followed by pooling reduce the sensibility to translation and rotation in the
input image.

If a well known architecture was chosen:
* What architecture was chosen?

The architecture is based on LeNet.

* Why did you believe it would be relevant to the traffic sign application?

LeNet works well for image recognition. It was created to recognize hand-written digits.
It seems like NN designed for image recognition generally use convolution and pooling.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The difference of accuracy between the test set and the validation set is low (95.6 vs 93.8%).
The accuracy on the validation test set is good (93.8%).

However the state of the art seems to give an accuracy of at least 98%.
There is room for improvement in my architeture.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The quality of the images is quite good. Their brigthness and contrast is diverse but I didn't
expect the model to have to much difficulty to give a good prediction.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| 60km/h     			| 60km/h 										|
| 80km/h				| 80km/h	    								|
| Children	      		| Children		    			 				|
| Bicycle    			| Bicycle           							|
| Ahead                 | Ahead                                         |
| Right                 | Right                                         |

The accuracy on the additional was 100%.
This was expected given the overall good quality of those images.
Some images from the initial test set have a much lower quality.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model is very confident for the prediction of most of the new images.

As expected the confidence levels are not as high (still very high) for the similar images.
Speed limit signs are very similar only a small part of the image is different between a 
30, 50, 60, 80km/h. Still the model prediction is accurate by a large margin. 


