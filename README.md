# MNIST Classifier from Scratch in C++

A simple, from-scratch implementation of a neural network for classifying handwritten digits from the MNIST dataset, built entirely in C++ without external machine learning libraries.

## Overview

This project implements a complete machine learning pipeline for handwritten digit recognition using only standard C++ and OpenMP for parallelization. The neural network is built from the ground up, including:

- Matrix operations and linear algebra
- Fully-connected neural network architecture
- Backpropagation algorithm
- Activation functions (ReLU and Softmax)
- Training and evaluation routines

The implementation demonstrates core machine learning concepts while achieving competitive performance on the classic MNIST benchmark.

##  Architecture

### Neural Network Structure
- **Input Layer**: 784 neurons (28×28 pixel images)
- **2 Hidden Layers**: 512 and 256 neurons
- **Activations**: ReLU and Softmax
- **Dropout Layers**: Regularization
- **Output Layer**: 10 neurons (digits 0-9)
- **Loss Function**: Cross-entropy

## Dataset

This project uses the **MNIST Database of Handwritten Digits**:
- **60,000 training images** + **10,000 test images**
- 28×28 pixel grayscale images
- 10 classes (digits 0-9)

Reference: [Yann LeCun's MNIST Page](http://yann.lecun.com/exdb/mnist/)

## Performance

Accuracy on the test dataset: **98.34%**

## License

This project is licensed under the MIT License - see the LICENSE file for details.