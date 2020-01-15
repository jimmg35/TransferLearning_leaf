# TransferLearning_leaf
This project demonstrates Transfer Learning based Convolutional Neural Network using the leaf dataset found at:
https://archive.ics.uci.edu/ml/datasets/leaf

Ther are several classes of leaf, but I only employ three types of leaf to be classified.<br>
Such as Aesculus Californica, Erodium Sp and Magnolia Grandiflora<br>

<img width="181" height="184" src="https://github.com/jimmg35/TransferLearning_leaf/blob/master/dataset/Train/AesculusCalifornica_04.JPG"> .
<img width="181" height="184" src="https://github.com/jimmg35/TransferLearning_leaf/blob/master/dataset/Train/ErodiumSp_03.JPG"> .
<img width="181" height="184" src="https://github.com/jimmg35/TransferLearning_leaf/blob/master/dataset/Train/MagnoliaGrandiflora_02.JPG"> .

<br>

Each class contains 7 images for training data, and 3 images for testing data.
You might wonder that the amount of images is not enough for training. Therefore, I applied Imagedatagenerator faking more than 1000 times data than original to this project.

# Implementation

The concept of Transfer Learning is that we take a pre-trained model such as ResNet50, AlexNet...etc as a general feature extractor, and subtitute its output layer with our own classifier depending on the condition of the dataset.




