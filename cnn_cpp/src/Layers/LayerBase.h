/**
 * @file LayerBase.h
 *
 * This header defines the foundational classes for building and operating
 * neural network layers within a deep learning framework, particularly focusing
 * on forward and backward propagation functionalities.
 *
 * Classes included:
 *
 * 1. LayerBase:
 *    - Abstract base class for all types of network layers.
 *    - Defines common interfaces for forward and backward operations, setting
 *      learning parameters, gradient checking, and more.
 *    - Contains methods for managing learning rate and regularization factor (lambda),
 *      calculating L2 norm of weights for regularization purposes, and displaying
 *      debug information.
 *    - Offers functionality for perturbing weights or biases to facilitate gradient checking.
 *    - Utilizes Eigen library for matrix and vector operations, making it suitable for
 *      handling high-performance mathematical computations inherent in neural networks.
 *
 * 2. OutputLayerBase:
 *    - Abstract base class for output layers of a neural network.
 *    - Handles the forward propagation from the final hidden layer to the output layer
 *      and computes backward propagation errors based on the loss between the predicted
 *      and actual labels.
 *    - Supports common loss functions and performance metrics through virtual methods that
 *      can be overridden by specific output layer implementations such as Softmax or SVM layers.
 *    - Includes utility methods for determining the class predictions from network outputs.
 *
 * Enumerations:
 * 1. CheckGradType:
 *    - Enum for specifying the type of parameter (WEIGHT or BIAS) being checked during gradient checking.
 *
 * 2. OutputLayerType:
 *    - Enum for distinguishing between different types of output layers, currently supporting Softmax and SVM.
 *
 * This design aims to provide a flexible and extensible architecture for constructing
 * various types of neural network layers while maintaining consistency in how these layers
 * are managed and interact within larger networks.
 */

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "Structure.h"
#include "Optimizer/AllOptimizers.h"

#pragma once

using namespace Eigen;

namespace dl {

// Enumeration to specify the type of gradient check (either for weights or biases).
enum class CheckGradType {
	WEIGHT = 0,
	BIAS = 1
};

// Enumeration to specify the type of output layer (either Softmax or SVM).
enum class OutputLayerType {
	Softmax = 0,
	SVM = 1
};

// Abstract base class for a neural network layer.
class LayerBase {
public:
	virtual ~LayerBase() {}

	// Virtual function to perform the forward pass of the layer.
	virtual void Forward(const MatrixType& x, bool isTraining = true) = 0;

	// Virtual function to perform the backward pass of the layer.
	virtual void Backward(const MatrixType& dz) = 0;

	// Optional virtual functions to return the output image size.
	virtual int OutImageSize() { return 0; }

	// Optional virtual functions to return the output channels.
	virtual int OutChannels() { return 0; }

	//// Sets the learning rate and regularization parameter lambda for the layer.
	//virtual void SetLearningRateAndLambda(FloatType learningRate, FloatType lambda) {
	//	learningRate_ = learningRate;
	//	lambda_ = lambda;
	//}

	virtual void SetOpertimizerMethod(OptimizerMethod optimizerMethod, FloatType lr, FloatType lambda) {
		if (!optimizer_ || optimizerMethod_ != optimizerMethod) {
			if (optimizerMethod_ == OptimizerMethod::ADAM) {
				optimizer_ = std::make_shared<AdamOptimizer>(lr, 0.9f, 0.999f, 1e-8f, lambda);
			}
			if (optimizerMethod_ == OptimizerMethod::SGD) {
				optimizer_ = std::make_shared<SGDOptimizer>(lr, lambda);
			}
			if (optimizerMethod_ == OptimizerMethod::MSGD) {
				optimizer_ = std::make_shared<MSGDOptimizer>(lr, lambda);
			}
		}
		else {
			optimizer_->SetLambda(lambda);
			optimizer_->SetLearningRate(lr);
		}
		optimizerMethod_ = optimizerMethod;
	}

	virtual FloatType RegularizationTerm() {
		return 0.;
	}

	// Method for displaying debug information.
	virtual void ShowDebugInfo() {
	}

	// Method to perturb parameters by a small epsilon for numerical gradient checking.
	// This method is used to slightly modify the weights or biases of the network to estimate gradients numerically.
	// @param positionRate: A FloatType value between 0 and 1 indicating the relative position
	//                      within the weights or biases vector to perturb.
	// @param epsilon: The small change to apply to the parameter at the specified position.
	// @return bool: Returns true if the perturbation is successful, false otherwise.
	virtual bool PerturbForGradientCheck(FloatType positionRate, FloatType epsilon) {
		if (positionRate < 0. && positionRate >= 1.) {
			return  false;
		}

		if (checkGradType_ == CheckGradType::WEIGHT && W_.rows() * W_.cols() > 0) {
			int position = static_cast<int>(positionRate * W_.rows() * W_.cols());
			W_.data()[position] += epsilon;
			return true;
		}

		if (checkGradType_ == CheckGradType::BIAS && biases_.size() > 0) {
			int position = static_cast<int>(positionRate * biases_.size());
			biases_.data()[position] += epsilon;
			return true;
		}

		return false;
	}

	// Retrieves the gradient value at a specified position rate for gradient checking.
	// This method is used to fetch the gradient from the gradient matrices after a forward and backward pass.
	// @param positionRate: A FloatType value between 0 and 1 indicating the relative position 
	//                      within the gradient matrix or vector.
	// @return FloatType: The gradient value at the specified position.
	virtual FloatType GetGradient(FloatType positionRate) {
		if (positionRate < 0. || positionRate >= 1.) {
			return 0.;
		}

		if (checkGradType_ == CheckGradType::WEIGHT && W_.rows() > 0) {
			int position = static_cast<int>(positionRate * W_.rows() * W_.cols());
			int col = static_cast<int>(position / dW_.rows());
			int row = static_cast<int>(position % dW_.rows());
			std::cout << "Weight's row, col = " << row << ", " << col << "\n";
			return dW_(row, col);
		}
		if (checkGradType_ == CheckGradType::BIAS && biases_.size() > 0) {
			int position = static_cast<int>(positionRate * biases_.size());
			std::cout << "Bias's position = " << position << "\n";
			return dBiases_(position);
		}

		return 0.;
	}

public:
	MatrixType outputMat_;   // Output matrix of the layer.
	MatrixType dxMat_;       // Derivative of the loss with respect to input.
	MatrixType W_;           // Weight matrix of the layer.
	MatrixType dW_;          // Derivative of the loss with respect to the weight matrix.
	VectorType biases_;      // Bias vector.
	VectorType dBiases_;     // Derivative of the loss with respect to the biases.
	CheckGradType checkGradType_ = CheckGradType::WEIGHT;  // Type of gradient check to perform.
	OptimizerMethod optimizerMethod_ = OptimizerMethod::ADAM;
	std::shared_ptr<OptimizerBase> optimizer_; // optimizer
};

// Abstract base class for an output layer.
class OutputLayerBase {
public:
	virtual ~OutputLayerBase() {}

	// Forward pass computation for the output layer.
	virtual void Forward(const MatrixType& x) = 0;

	// Backward pass computation for the output layer.
	virtual void Backward(const MatrixType& softmaxOut, const VectorXi& labels) = 0;

	// Computes the loss given the predictions and true labels.
	virtual FloatType Loss(const MatrixType& pred, const VectorXi& label) = 0;

	// Returns the index of the maximum element in a vector (argmax operation).
	virtual int Argmax(const VectorType& x) {
		Eigen::Index maxId;
		x.maxCoeff(&maxId);
		return static_cast<int>(maxId);
	}

public:
	MatrixType outputMat_; // Output matrix after forward pass.
	MatrixType dxMat_;     // Gradient of the loss with respect to input.
};
}