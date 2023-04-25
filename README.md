# Predicting Types of vegetables based on Images

## Abstract

There is a significant demand to eat healthier but many people don't have the background in types of produce to ensure they are always using the correct ingredients. With this project, we aim to create a tool that can identify what a given vegtable is usingmachine learning techniques.

## Introduction

As people who would like to eat an adequate amount of vegetables, and as avid farmer's market enjoyers, we often found ourselves overwhelmed with the many different types of vegetables out there. 
Using an image dataset found on kaggle.com with 15 different types of vegetables, this is a well-suited situation for a convolutional neural network.
That being said, we had not used a dataset quite this large in class and had various concerns about whether a standard laptop would be able to handle the amount of processing power needed to train the model.
Initially, PyTorch was the most obvious library to use as we had the ost familiarity, though we initially had plans to try different libraries that were not deemed necessary as any method that attempts to train the model from scratch is likely to need the same significant amount of work.
The main limitations associated with training a model from scratch are the amount of processing power needed and we initially tried using a discovery cluster before moving on to a pre-trained model.


## Setup

The dataset used was conveniently already split up into a train, test, and validation set, with a folder for each vegetable - a total of 15. Each folder in the training set contained x images and each folder in the test and validation sets contained y images. 

## Results

We found much greater success in the tensorflow model than the PyTorch model, and were able to create a very accurate model using transfer learning.
This success can be seen in the following visuals:

![confusion matrix]()
Exhibit 1: Confusion matrix where higher numbers on the diagonal indicate correctly predicted images.

![loss]()
Exhibit 2

## Discussion

## Conclusion

## References

Dataset citation: https://www.researchgate.net/publication/352846889_DCNN-Based_Vegetable_Image_Classification_Using_Transfer_Learning_A_Comparative_Study

[Kaggle link](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset?resource=download)

