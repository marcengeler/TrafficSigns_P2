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
[crop_ratio]: ./examples/crop_ratio.PNG "Crop Ratio"
[own_images]: ./examples/own_images.PNG "Own Images"
[accuracy_result]: ./examples/accuracy.PNG "Accuracy"
[skew]: ./examples/skew.PNG "Image Skewing"
[balance2]: ./examples/balancing_2.PNG "Class Balance"
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

To check the dataset i chose random 5 images from each class to check if this looks good. Example:

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

To further enhance the image data, without loss of information, the images were skewed up to 10° degrees

![alt text][skew]

In order for the images, to stay as close to the originals as possible, i had to make sure, that the exrapolation methods at the borders, for all those applications were set to encorporate the surrounding pixels. OpenCV offers this functionality
and I had to make sure to include this. After these new datapoints, the balancing issue could also be solved more or less. I capped the specific classes at 3 * the images the smalles class has. 

![alt text][balance2]

### Overfitting Testing

The simplest way for me to test overfitting, was to plot training and test results over all iterations. If there was the slightest amount of overfitting, a decrease in validation accuracy would show up in the plot. I was confident, that
with the result below, my system fit the general case quite well.

![alt text][accuracy]

### Test a Model on New Images

I have chosen 5 traffic signs found on the web. Based on the training data we had, it was necessary to not crop too closely or too far away to the image, or it wouldn't detect the traffic sign correctly. The first five examples show the accuracy of the system, while the
last one shows the issue with taking the right cropping ratio.

![alt text][own_images]

#### Prediction on new Images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign  (100%)								| 
| Go straight or right	| Go straight or right (100%)					|
| 70					| 70 (100%)										|
| Roundabout      		| Keep Left	(100%)				 				|
| No Entry				| No Entry	(100%)    							|


The model could correctly  guess 4 of the first 5 images provided, which translates to 80% test accuracy. However, with the last image, it shows it's downsides if the crop ratio of the image is not properly set.

![alt text][crop_ratio]
