# PyTorch C++ API test a small convolution neural network on Kaggle Flowers datasets.

Train and Test a Deep neural network with PyTorch C++ API Libtorch on 5 classes of flowers get from Kaggle Flowers dataset.

![](5_classes_flowers_tulips.png)

![](5_classes_flowers_rose.png)

## Installation guide on Ubuntu machine:

#### Directly installation:

https://github.com/ollewelin/torchlib-opencv-gpu

#### Or use Anaconda enviroment:

https://github.com/ollewelin/Installing-and-Test-PyTorch-C-API-on-Ubuntu-with-GPU-enabled

## Download and build this repository:

### Get this repo

    $ mkdir class_test
    $ cd class_test
    $ git clone https://github.com/ollewelin/5_classes_PyTorch_test
    $ cd 5_classes_PyTorch_test
    
### Open in Visual Code

    $ code .

### Change CMakeLists.txt to fit your path

    list(APPEND CMAKE_PREFIX_PATH "/home/olle/class_test/5_classes_PyTorch_test/libtorch")

## Make a dataset with 5 categories 


### Download flowers datasets from kaggle

https://www.kaggle.com/alxmamaev/flowers-recognition

### Make 5 data folders with 503 jpg files abriatary size images

### Data structure

    repo_root 
    
    CMakeLists.txt
    file_names.csv
    main.cpp
    model.h
    
    repo_root/libtorch
    
    repo_root/data
    repo_root/data/class0/0..502.jpg files
    repo_root/data/class1/0..502.jpg files
    repo_root/data/class2/0..502.jpg files
    repo_root/data/class3/0..502.jpg files
    repo_root/data/class4/0..502.jpg files


