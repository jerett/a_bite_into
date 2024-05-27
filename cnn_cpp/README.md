# Simple CNN CPP Framework

This project is a simple CNN (Convolutional Neural Network) framework implemented in C++ using Eigen and OpenCV libraries. 
It provides a basic setup for building and training neural networks, including various common layers and optimization techniques. 
The primary goal of this framework is to demonstrate fundamental concepts and operations of CNNs, 
making it an educational tool for understanding and experimenting with neural networks.

## Project Structure

- `src/`: Contains the core implementation of the framework.
- `samples/`: Contains various demos demonstrating the capabilities of the framework.
- `data/`: Contains datasets used by the demos in the `samples/` directory. You need to copy the data dir to the build binary dir(or change data path).

### src/
The `src/` directory includes the implementation of the following components:
- **Layers**: Convolutional layer, Fully connected layer, ReLU layer, Softmax layer, SVM layer, Maxpooling layer, Batch Normalization layer.
- **Optimizers**: SGD, MSGD (Momentum SGD), and Adam.
- **Model Class**: A basic class for constructing and managing networks.

### samples/
The `samples/` directory includes several demos:
- Binary classification on synthetic data.
- MLP network on MNIST dataset.
- CNN network on MNIST dataset.
- CNN network on CIFAR-10 dataset.
- Gradient checking.
### samples/
The `samples/` directory includes several demos demonstrating the capabilities of the framework:
- **Binary Classification on Synthetic Data**: This demo showcases a simple MLP model performing binary classification. 
    It is designed to demonstrate the basic functionality of constructing and training a network to distinguish between two categories using artificially generated data.
	
- **MLP Network on MNIST Dataset**: This demo utilizes a Multi-Layer Perceptron (MLP) for recognizing handwritten digits from the MNIST dataset. 
    It illustrates how to use the framework for typical image classification tasks.
	
- **CNN Network on MNIST Dataset**: Demonstrates the application of a Convolutional Neural Network (CNN) to the MNIST dataset.
    This demo provides insight into how CNNs can be effectively used for image recognition tasks within this framework.
	
- **CNN Network on CIFAR-10 Dataset**: This demo applies a CNN to classify images from the CIFAR-10 dataset, 
    showcasing the framework's ability to handle more complex image classification scenarios involving color images and multiple classes.
	
- **Gradient Checking**: This demo allows for the verification of gradient computations within the network. 
    It is advisable to use double precision to minimize the impact of numerical precision issues on the gradient results. 
	Users are recommended to disable or minimize the use of non-differentiable or highly non-linear layers such as ReLU, 
	Batch Normalization, and MaxPooling during gradient checking to ensure accurate verification.

### data/
Contains the datasets used in the demos. Each demo has its respective data organized within this folder.

## Features

- **Multiple Layer Types**: Includes essential layers like Convolutional, Fully Connected, ReLU, Softmax, SVM, and Batch Normalization.
- **Optimization Methods**: Supports basic and advanced optimizers like SGD, Momentum SGD, and Adam.
- **Precision Options**: The framework can be compiled for either float or double precision. Double precision is recommended for checking the correctness of gradient propagation.
- **Demos**: Ready-to-run demos illustrating how to use the framework for common tasks and datasets.

## Data Augmentation Techniques

To enhance the generalization ability of the neural network models, especially in the CNN demos, the framework incorporates various data augmentation techniques. 
These techniques help in simulating a more diverse set of training data, thus improving model robustness. The following data augmentation strategies are implemented:

- **Horizontal Flipping**: Randomly flips images horizontally, mirroring data along the vertical axis. This is particularly useful for datasets where orientation is not a factor.
- **Random Scaling**: Adjusts the size of images by a certain factor randomly chosen within a predefined range, helping the model to learn scale invariance.
- **Random Gamma Correction**: Applies random gamma corrections to alter the luminance of images, simulating different lighting conditions.
- **Random Cropping**: Randomly crops regions from images, forcing the model to focus on different parts of the input data.

These augmentation techniques are configurable and can be applied individually or combined to enhance training sessions for the CNN models.

## Educational Purpose

This framework is designed primarily for understanding the basics of neural networks and convolutional neural networks. 
It is not intended for production environments or to be used as an industrial-strength solution. 
The implementations focus on clarity and simplicity rather than on optimization and performance.

## Limitations

While this framework demonstrates the core concepts of CNNs and includes implementations of various neural network components:
- **It is not optimized for high performance and scalability** necessary for real-world applications.
- **Lacks robust error handling, logging, and other features expected in production-level code**.

## Getting Started

To get started with this framework, clone the repository and navigate to the respective directories to run the demos.

### Prerequisites

Ensure you have Eigen and OpenCV installed on your system.

### Compilation and Running

Navigate to the `src/` directory and compile the source files using your preferred C++ compiler, linking against Eigen and OpenCV libraries. Follow the instructions in the `samples/` directory for running specific demos.
