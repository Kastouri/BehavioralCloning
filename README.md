# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

            
[model_summary]: ./writeup/model_summary.png "Model Visualization"
[center]: ./writeup/center.jpg "Image from the centre"
[recovering]: ./writeup/recovering.gif "Recovering"
[flipping]: ./writeup/flipping.png "Flipping Images"
[mse]: ./writeup/mse.png "Mean Squarred Error"
[track_2_good]: ./writeup/track_2_good.gif "Driving in the Second Track"
[track_2_bad]: ./writeup/track_2_bad.gif "Driving in the Second Track"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 5 convolution layers and 3 fully connected layers. My model was inspired by nvidia's model that was introduced in the course. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer so that the inputs are before -0.5 and 0.5. 

#### 2. Attempts to reduce overfitting in the model

After each layer I added a dropout layer with a dropout probability of 0.25 (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was set to 0.001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and couter clockwise driving. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with start with a small model and between increasing the model size and testing it in the simulation.

My first step was to use a convolution neural network model with a couple of convolutin layers similar to the models introduced in the lessons. I thought this model might be appropriate because although it was simple it performed relatively well in the better part of the track.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by introducing dropout and maxpooling layers.

Then I collected more data to ensure that the model is not overfitting any more.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I collected data 
while taking sharp turns and recovering from the sides of the road. I trying to drive around the sharp turns slowly and using the mouse as a controller. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes 
![alt text][model_summary]


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to get back to the center of the track. These images show what a recovery looks like starting from right side of the track :

![alt text][recovering]


Then I repeated this process on the same track driving counter clockwise in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would correct the models bias to turning to the left. For example, here is an image that has then been flipped:

![alt text][flipping]

After the collection process, I had 29540 number of images from the centre, 29540 from the left and 29540 from the right. I then preprocessed this data by flipping it.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 as evidenced by the figure below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][mse]

Although trained with images **only** from the first track, the model performed well in some parts of the second track. The next animation shows an example:

![alt text][track_2_good]

But clearly data from the second track is needed to achieve better performance. Here is an example where my model fails
![alt text][track_2_bad]


