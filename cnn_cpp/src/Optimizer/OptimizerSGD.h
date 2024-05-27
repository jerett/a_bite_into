#include <Eigen/Dense>
#include "OptimizerBase.h"
#pragma once

using namespace Eigen;

namespace dl {
class SGDOptimizer : public OptimizerBase {
public:
    SGDOptimizer(FloatType learningRate, FloatType lambda) {
        learningRate_ = learningRate;
        lambda_ = lambda;
    }

    void Update(MatrixType& W, MatrixType& dW, VectorType& b, VectorType& db) override {
        W -= (dW + W * lambda_) * learningRate_;
        b -= db * learningRate_;
    }
};

class MSGDOptimizer :public OptimizerBase {
public:
    MSGDOptimizer(FloatType learningRate, FloatType lambda) {
        learningRate_ = learningRate;
        lambda_ = lambda;
    }

    void Update(MatrixType& W, MatrixType& dW, VectorType& b, VectorType& db) override {
        if (mdW_.size() == 0) {
            mdW_ = MatrixType::Zero(dW.rows(), dW.cols());
            mdB_ = VectorType::Zero(db.rows());
        }

        mdW_ = mdW_ * momentum_ + dW * (1 - momentum_);
        mdB_ = mdB_ * momentum_ + db * (1 - momentum_);

        W -= (mdW_ + W * lambda_) * learningRate_;
        b -= mdB_ * learningRate_;
    }

private:
    MatrixType mdW_;
    VectorType mdB_;
    FloatType momentum_ = 0.9f;
    int t_ = 0;
};

}