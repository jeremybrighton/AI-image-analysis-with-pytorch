1
Get the GPU Device

Store the device name in the variable device.


  2
Find the names of the classes

Find the names of the classes we will be working with. Images for each class are inside folders within the sea_creatures/train folder. Make a list of the class names (each corresponding to a directory name). Your list should be named classes.


  3
Build a transformer pipeline

Build a transformer pipeline. It should ensure the images are in RGB format, scale them to 224×224 pixels, and convert them into a PyTorch tensor. You will probably find ConvertToRGB useful.

4
Test that the transformer pipeline is working

Test that the transformer pipeline is working. Load in the specified training image and transform it. Check that you get a 3x224x224 tensor.


5
Create the training data DataSet

Create a DataSet for the training data (using the ImageFolder subclass). It should apply the transformer pipeline.

 6
Calculate the class distribution

Calculate the class distribution. Store this in the variable class_distribution as a dictionary. The keys should be the class names. The values should be the number of training samples for the class.

7
Create a DataLoader

Create a DataLoader that loads from this DataSet in batches of 32.

8
Calculate the mean and standard deviation of each channel in this data set

Fill the missing lines in the get_mean_std function and invoke it with the right dataset.

This will calculate the correct values for mean and std.


9
Build a transformer pipeline that normalizes the channels

Build a new transformer pipeline that normalizes the channels according to the mean and standard deviation above. The pipeline should be assigned to the variable transform_norm. Afterwards, use the pipeline to create a normalized data set and store it in the variable norm_dataset.


10
Split the normalized dataset

Split the normalized data set into a training set and a validation set. 80% of the data should be in the training set, and 20% in the validation set.


11
Set up dataloaders for the normalized datasets

Set up data loaders for both the training and validation data sets. Use the same batch size as before. Remember to set shuffle=True on the training loader.


Build the Model

12
Start setting up the network

Start setting up the network. We'll begin with three layers: - 2D convolution with sixteen 3x3 kernels - ReLU activation - Max pooling with 4x4 kernels (and a stride of 4x4)

Expected outcome:

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [32, 16, 56, 56]          --
├─Conv2d: 1-1                            [32, 16, 224, 224]        448
├─ReLU: 1-2                              [32, 16, 224, 224]        --
├─MaxPool2d: 1-3                         [32, 16, 56, 56]          --
==========================================================================================
Total params: 448
Trainable params: 448
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 719.32
==========================================================================================
Input size (MB): 19.27
Forward/backward pass size (MB): 205.52
Params size (MB): 0.00
Estimated Total Size (MB): 224.79
==============================================================


13
Add three more layers to the network

2D convolution with thirty-two 3x3 kernels
ReLU activation
Max pooling with 4x4 kernels

14
Add four more layers to the network

2D convolution with sixty-four 3x3 kernels
ReLU activation
Max pooling with 4x4 kernels
Flattening


15
Add the final layers to the model

Drop-out (with p=0.5)
Linear layer with 500 outputs (check the summary above to get the correct number of inputs)
ReLU activation
Drop-out
Linear output layer with the appropriate number of output


16
Prepare for training

Prepare for training. Define the loss function, create an optimizers, and send the model to the GPU device.


17
Train model for 10 epochs.

Import the train and predict functions from training.py and run the training process for 10 epochs.


                                      Evaluate the Model Performance
18
Make predictions for all of the images in the validation set

Make predictions for all of the images in the validation set. Start by calculating the probabilities using the predict function (from our training.py module). And then calculate the predicted class based in the probabilities.


19
Create the confusion matrix

Create the confusion matrix for the predictions on the validation set. We have provided the actual classes in the targets variable.

    
20
Create a data set for the test data

Create a data set for the test data. It is located in the sea_creatures/test directory. Then create a data loader from this data set. DO NOT shuffle this data!

21
Make a prediction for each of the test images

Correct
