#include <Eigen/Dense>
#include "LayerBase.h"
#pragma once

using namespace Eigen;

namespace dl {
// As you see.
class ReLU :public LayerBase {
public:
	ReLU() {}

	virtual void Forward(const MatrixType& x, bool isTraining) override {
		outputMat_ = x;
		dxMat_.resize(x.rows(),x.cols());
		dxMat_.setOnes();
		for (int i = 0; i < outputMat_.rows(); ++i) {
			for (int j = 0; j < outputMat_.cols(); ++j) {
				if (outputMat_(i, j) < 0) {
					outputMat_(i, j) *= static_cast<FloatType>(0.01);
					dxMat_(i, j) = static_cast<FloatType>(0.01);
				}
			}
		}
	}

	virtual void Backward(const MatrixType& dz) override {
		dxMat_.array() *= dz.array();
	}

public:
	VectorType output_;
	VectorType dzIn_;
	MatrixType x_;
};

}