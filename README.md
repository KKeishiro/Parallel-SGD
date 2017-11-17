# Parallel-SGD
Image classification by Parallel SGD with Random Fourier Features

## Description
The purpose of this project is to build a classification model to classify images in one of two classes according to their visual content. 
Given labeled dataset contains two sets of images: **Portraits** and **Landscapes**.

## Dataset
A set of 400 features has been extracted from each picture. 
16k training images (**handout_train.txt**) and 4k testing images (**handout_test.txt**), sampled rougly at the same frequency from both categories. 
Provided data only has the features for each image, from which the actual image cannot be reconstructed. Each line in the files corresponds to one image and is formatted as follows:

1. Elements are space separated.
2. The first element in the line is the class y {+1,-1} which correspond to Portrait and Landscape class, respectively.
3. The next 400 elements are real numbers which represent the feature values x0... x399.
