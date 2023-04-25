# Predicting Types of vegetables based on Images

## Abstract

There is a significant demand to eat healthier but many peopledon't have the background in types of produce to ensure they are always using the correct ingredients. With this project, we aim to create a tool that can identify what a given vegtable is usingmachine learning techniques.

## Introduction

As people who would like to eat an adequate amount of vegetables, and as avid farmer's market enjoyers, we often found ourselves overwhelmed with the many different types of vegetables out there. Using an image dataset found on kaggle.com with 15 different types of vegetables, this is a well-suited situation for a neural network. Since this issuch a large dataset, we know that training a neural network from scratch would take a significant amount of time, 

## Setup

a) The dataset used was conveniently already split up into a train, test, and validation set, with a folder for each vegetable - a total of 15. Each folder in the training set contained 1000 images, each folder in the test and validation sets contained 200 images. The folders are filled of each vegetable either alone or multiple with different backgrounds.

b) We ran and created our models on our local computers. Our best model consisted of using tranfer learning from the Inception version 3 model. The InceptionV3 model is a deep learning model that has been trained on a large dataset of images called ImageNet, which contains over 14 million images across more than 20,000 classes. The model is known for its ability to classify images with high accuracy while minimizing the number of parameters required, and was choosen because it can identify certain features of the vegetables that may be common across different types and use them to make more accurate classifications and may have been trained on some images of vegetables as well. Since this model was already very accurate and large we did not wish to retrain this model but rather use this model's output and add on more convonlutional layers instead. As for parameters we just put in the imahges since the  inception model is built to be able to process phhotts efficently already. 

## Results

[confusion matrix](https://github.com/asodergren/vegetables/blob/main/confusion_matrix.png)

## Discussion

## Conclusion

## References

Dataset citation: https://www.researchgate.net/publication/352846889_DCNN-Based_Vegetable_Image_Classification_Using_Transfer_Learning_A_Comparative_Study
<<<<<<< HEAD

[Kaggle link](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset?resource=download)

test
=======
>>>>>>> eff43d640bf8a34b9a5f7762e819b9d0afcc7dda
