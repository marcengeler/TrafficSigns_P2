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

[balance]: ./examples/chart_class_num.PNG "Class Balance"
[sanity]: ./examples/sanity_checks.PNG "Sanity Checks"
[lenet2]: ./examples/yann_lecun.PNG "New Network Architecture"
[shifting]: ./examples/shifting.PNG "Image Shifting"
[scaling]: ./examples/scaling.PNG "Image Scaling"
[brightness]: ./examples/brightness.PNG "Image Brightness Adjust"
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points

---

### Data Set Summary & Exploration

#### Class Balance

The first subject I addressed in the dataset was the class balance. To get a good training dataset i corrected the amount of different classes which were available to the minimum amount in all classes. Which resultet in arount 175 teaching images
per class. This is a bit low, but it was a good starting point to get a balanced dataset.
Balanced datasets are really important, if we want to avoid overfitting a certain class. As you can see in the plot there are certain classes which are highly overrepresented. And could lead to a skewed result in the training which favors
those overrepresented classes.

![alt text][balance]

#### Normalization

To enhance the training process and to improve the result, the input data was normalized to have roughly mean 0 and variance 1. To do this, the input pixel values were normalized as follows

x / 128 - 128

which roughly normalized an image containing values from 0 to 255.

#### Visualization and Sanity Checks

To check the dataset i chose random 5 images from each class to check if this looks good

![alt text][sanity]

#### Dataset Metrics

To get further insight into the dataset, i looked at the arrays provided, to get the following information

Number of training examples = 34799, reduced to 7697 after class balancing
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43


### Design and Test a Model Architecture

For this exercise I decided against using grayscale imaging, because the rgb values can hold a lot of information for traffic signs. Especially the red color can be revealing, what type of traffic sign it is to the human eye.
Because there is so much information for humans stored with color, I presumed, that this information has to be kept in order for the neural network to leverage this information as well.

However, as mentioned before I normalized the data input.

#### First Model: LeNet Architecture with Larger Layers

My first model was the LeNet Architecture with some larger layers to amend the amount of classes compared to the emnist dataset leNet was trained on. I used this architecture because there is a lot of information on it, and there were
some positive results already. This first show was to learn how well performing the network can be.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5 x 5   	| 1 x 1 stride, valid padding, outputs 32x32x18	|
| RELU					|												|
| Max pooling	      	| 2 x 2 stride,  outputs 16x16x18 				|
| Convolution 5 x 5	    | 1 x 1	stride, valid padding, outputs 16x16x48 |
| RELU					|												|
| Max pooling	      	| 2 x 2 stride,  outputs 8x8x48 				|
| Fully connected		| 360 Hidden Units								|
| Fully connected		| 180 Hidden Units								|
| Output Layer			| 43 Classes									|
| Softmax				|    	    									|
 
##### Training

To train the model, i used 70 Epochs and Batch Sizes of 128, as well as a learning rate of 0.001 without any decay. The loss function was computed using a cross entropy loss computed with the AdamOptimizer.

##### Result

Accuracy, 70%

#### Second Model: Own Architecture

Because the first Model was not able to reach an accuracy of above 90% i decided to take a look at literature. The Sermanet/LeCunn Article showed promising results on grayscale images, thus I adapted the input arrays to grayscale.
Also, the network had a different architecture and is depicted here.

![alt text][lenet2]

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale Image						| 
| Convolution 5 x 5   	| 1 x 1 stride, valid padding, outputs 32x32x18	|
| RELU					|												|
| Max pooling	      	| 2 x 2 stride,  outputs 16x16x18 				|
| Convolution 5 x 5	    | 1 x 1	stride, valid padding, outputs 16x16x48	|
| RELU					|												|
| Max pooling	      	| 2 x 2 stride,  outputs 8x8x48 				|
| Convolution 5 x 5	    | 1 x 1	stride, valid padding, outputs 8x8x600	|
| RELU					|												|
| Concatenation of Layer 2 and 3	|									|
| Dropout Layer			| Keep Prob = 0.5								|
| Fully connected		| 1800 Hidden Units								|
| Output Layer			| 43 Classes									|
| Softmax				|    	    									|

In the first shot I reached a validation accuracy of around 94%. However, this is not close to the result  mentioned in the paper. So I decided to apply some data augmentation on the image, to increase the training set size to an acceptable size.

##### Data Augmentation Methods

The first method I used, was switching to grayscale image, als in the publication from Sermanet/LeCunn, this proved to be a good increase in accuracy. Also I introduced some data augmentation.

The first method, which was introduced, was the random shifting of the image. This way, the image was shifted by up to two pixels in a random direction, this doubled the number of available images from ~7000 to over 14000 training images. However,
this was still not enough to get good and reliable training results.

![alt text][shifting]

Another method which was implemented was random image scaling. For any image in the test set, the image was also randomly scaled in addition to the original image.

![alt text][scaling]

Another method which was implemented was random brightness adjustment.

![alt text][brightness]

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

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
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


