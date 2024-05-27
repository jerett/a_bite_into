#include <Eigen/Dense>
#include "LayerBase.h"
#pragma once

using namespace Eigen;

namespace dl {
class Softmax :public OutputLayerBase {
public:
	Softmax(){}

	void Forward(const MatrixType& x) override {
		outputMat_ = x;
		for (int j = 0; j < x.cols(); ++j) {
			outputMat_.col(j) = SoftmaxForward(x.col(j));
		}
	}

	// Computes the gradients of the softmax outputs with respect to the loss.
	void Backward(const MatrixType& softmaxOut, const VectorXi& labels) override {
		dxMat_ = softmaxOut;
		for (int j = 0; j < softmaxOut.cols(); ++j) {
			dxMat_(labels[j], j) -= 1.;
		}
	}

	FloatType Loss(const MatrixType& pred, const VectorXi& label) override {
		FloatType dataTerm = 0.;
		for (int j = 0; j < pred.cols(); ++j) {
			FloatType p = pred.col(j)(label(j));
			p = std::max(FloatType(1e-20), p);
			dataTerm += -log(p);
		}
		return dataTerm;
	}

public:
	// Computes the softmax activation function for a single vector.
	VectorType& SoftmaxForward(const VectorType& x) {
		output_ = x;
		FloatType maxVal = output_(0);
		for (int i = 1; i < output_.size(); ++i) {
			maxVal = std::max(maxVal, output_(i));
		}

		FloatType sum = 0.;
		for (int i = 0; i < output_.size(); ++i) {
			FloatType v = output_(i) - maxVal;
			v = exp(v);
			output_(i) = v;
			sum += v;
		}
		output_ *= 1 / sum;
		return output_;
	}

public:
	VectorType output_;
};

}