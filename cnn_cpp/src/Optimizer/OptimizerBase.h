#include <Eigen/Dense>
#include "Structure.h"
#pragma once

using namespace Eigen;

namespace dl {
enum class OptimizerMethod {
	SGD = 1,
	MSGD = 2,
	ADAM = 3
};

class OptimizerBase {
public:
	virtual ~OptimizerBase() {

	}

	virtual void Update(MatrixType& W, MatrixType& dW, VectorType& b, VectorType& db) = 0;

	virtual void SetLambda(FloatType lambda) {
		lambda_ = lambda;
	}

	virtual void SetLearningRate(FloatType lr) {
		learningRate_ = lr;
	}

	FloatType Lambda() {
		return lambda_;
	}

protected:
	FloatType learningRate_;
	FloatType lambda_;
};

}