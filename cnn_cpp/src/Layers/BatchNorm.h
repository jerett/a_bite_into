#include <Eigen/Dense>
#include "LayerBase.h"

#pragma once
using namespace Eigen;

namespace dl {
/**
 * @brief Batch normalization layer for deep learning models.
 *
 * The BatchNorm class implements batch normalization, a technique used in deep learning
 * to normalize the activations of each layer. This helps stabilize and accelerate training
 * by reducing internal covariate shift.
 */
class BatchNorm : public LayerBase {
public:
	BatchNorm() {
	}

	virtual void Forward(const MatrixType& x, bool isTraining) {
		// Initialize parameters if not already initialized
		if (W_.rows() != x.rows() || W_.cols()!=1) {
			int rows = static_cast<int>(x.rows());
			W_ = VectorType::Ones(rows);
			biases_ = VectorType::Zero(rows);
			movingMean_ = VectorType::Zero(rows);
			movingVar_ = VectorType::Ones(rows);
			momentum_ = 0.9f;
		}

		inputMat_.resize(x.rows(), x.cols());
		inputMat_.noalias() = x;

		// Compute mean and variance during training
		if (isTraining) {
			mean_.resize(x.rows());
			var_.resize(x.rows());
			for (int i = 0; i < x.rows(); ++i) {
				mean_(i) = x.row(i).mean();
				var_(i) = (x.row(i).array() - mean_(i)).square().mean();
			}

			// Update moving averages of mean and variance
			movingMean_ = momentum_ * movingMean_ + (1 - momentum_) * mean_;
			movingVar_ = momentum_ * movingVar_ + (1 - momentum_) * var_;
		}
		else {
			// Use moving averages during inference
			mean_ = movingMean_;
			var_ = movingVar_;
		}

		// Compute normalized output using mean, variance, weights, and biases
		outputMat_.resize(inputMat_.rows(), inputMat_.cols());
		for (int i = 0; i < outputMat_.rows(); ++i) {
			FloatType gamma_x_f2 = static_cast<FloatType>(W_(i) * 1. / sqrt(var_(i) + 1e-10));
			outputMat_.row(i).array() = (inputMat_.row(i).array() - mean_(i)) * gamma_x_f2 + biases_(i);
		}
	}

	virtual void Backward(const MatrixType& dY) {
		dxMat_.resize(dY.rows(), dY.cols());
		if (dW_.rows() != W_.rows()) {
			dW_ = VectorType::Zero(W_.rows());
			dBiases_ = VectorType::Zero(W_.rows());
		}

		auto N = inputMat_.cols();
		xRow_.resize(N);
		dxRow_.resize(N);
		dzRow_.resize(N);
		for (int row = 0; row < inputMat_.rows(); ++row) {
			dzRow_.noalias() = dY.row(row);
			xRow_.noalias() = inputMat_.row(row);

			// Compute gradients for backpropagation
			// f = f1 * f2, where f1 = (x-mean), f2 = 1/sqrt(cov+1e-5)
			FloatType f2 = static_cast< FloatType>(1. / sqrt(var_(row) + 1e-10));
			FloatType a = -f2 * f2 * f2 / N;
			FloatType dzSum = dzRow_.sum();
			FloatType tmpa = static_cast<FloatType>(f2 * (-1. / N) * dzSum);
			FloatType tmpb = ((dzRow_.array() * (xRow_.array() - mean_(row)))).sum();
			FloatType tmpc = tmpa - a * mean_(row) * tmpb;
			FloatType atmpb = a * tmpb;

			for (int j = 0; j < dxRow_.size(); ++j) {
				dxRow_(j) = tmpc + f2 * dzRow_(j) + atmpb * xRow_(j);
			}

			FloatType dgamma = f2 * tmpb * N;
			FloatType dbeta = dzSum * N;

			dxMat_.row(row) = dxRow_ * W_(row);
			dW_(row) = dgamma;
			dBiases_(row) = dbeta;
		}

		MatrixType dwUpdate = dW_ / N;
		VectorType dbUpdate = dBiases_ / N;
		optimizer_->Update(W_, dwUpdate, biases_, dbUpdate);
	}


public:
	MatrixType inputMat_;
	FloatType momentum_;
	VectorType mean_, var_;
	VectorType movingMean_, movingVar_;
	VectorType xRow_, dxRow_, dzRow_;
};
}