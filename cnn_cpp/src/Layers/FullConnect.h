#include <Eigen/Dense>
#include "LayerBase.h"

#pragma once

using namespace Eigen;

namespace dl {
/**
 * @brief Fully connected layer class.
 *
 * This class represents a fully connected layer in a neural network.
 * It connects every neuron in the input layer to every neuron in the output layer.
 */
class FullConnect :public LayerBase {
public:
	// Constructor for FullConnect class.
	// @param outputSize The number of neurons in the output layer.
	// @param learningRate The learning rate used for weight updates during training.
	// @param lambda The regularization parameter.
	FullConnect(int outputSize) {
		outputSize_ = outputSize;
	}

	// As you see.
	virtual void Forward(const MatrixType& x, bool isTraining) override {
		if (W_.rows() != outputSize_ || W_.cols() != x.rows() + 1) {
			FloatType stddev = sqrt(2.0 / (x.rows() + 1));
			W_ = MatrixType::Random(outputSize_, x.rows() + 1) * stddev;
			W_.rightCols(1).setZero();
			dW_ = MatrixType::Zero(W_.rows(), W_.cols());
		}

		if (xWithBiasMat_.rows() != x.rows() + 1 || xWithBiasMat_.cols() != x.cols()) {
			xWithBiasMat_.resize(x.rows() + 1, x.cols());
			xWithBiasMat_.setOnes();
		}
		xWithBiasMat_.topRows(x.rows()) = x;

		outputMat_.resize(W_.rows(), x.cols());
		outputMat_.noalias() = W_ * xWithBiasMat_;
	}

	// As you see.
	virtual void Backward(const MatrixType& dz) override {
		dW_.noalias() = dz * xWithBiasMat_.transpose();

		dxMat_.resize(W_.cols() - 1, dz.cols());
		dxMat_.noalias() = W_.leftCols(W_.cols() - 1).transpose() * dz;

		if (dz.cols() > 0) {
			dwUpdate_ = dW_ / dz.cols();
			MatrixType weights = W_.leftCols(W_.cols() - 1);
			VectorType bias = W_.rightCols(1);
			MatrixType dweights = dwUpdate_.leftCols(dW_.cols() - 1);
			VectorType dbias = dwUpdate_.rightCols(1);

			optimizer_->Update(weights, dweights, bias, dbias);

			W_.leftCols(W_.cols() - 1) = weights;
			W_.rightCols(1) = bias;
		}
	}

	virtual FloatType RegularizationTerm() override {
		if (W_.rows() > 0 && W_.cols() > 1) {
			return W_.leftCols(W_.cols() - 1).norm() * 0.5 * optimizer_->Lambda();
		}
		return 0.;
	}

public:
	MatrixType xWithBiasMat_;
	int outputSize_;
	MatrixType dwUpdate_;
};

}

