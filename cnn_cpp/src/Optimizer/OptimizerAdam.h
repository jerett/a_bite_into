#include "OptimizerBase.h"
#include <Eigen/Dense>
#pragma once
using namespace Eigen;

namespace dl {
class AdamOptimizer : public OptimizerBase {
public:
    AdamOptimizer(FloatType alpha = 1e-3, FloatType beta1 = 0.9, FloatType beta2 = 0.999, FloatType epsilon = 1e-8, FloatType lambda = 5e-3)
        : beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {
        learningRate_ = alpha;
        lambda_ = lambda;
    }

    void Update(MatrixType& weights, MatrixType& dWeights, VectorType& biases, VectorType& dBias) override {
        t_ += 1;

        if (mWeights_.size() == 0) {
            mWeights_ = MatrixType::Zero(dWeights.rows(), dWeights.cols());
            vWeights_ = MatrixType::Zero(dWeights.rows(), dWeights.cols());
            mBiases_ = VectorType::Zero(dBias.size());
            vBiases_ = VectorType::Zero(dBias.size());
        }

        mWeights_ = beta1_ * mWeights_ + (1 - beta1_) * dWeights;
        vWeights_ = beta2_ * vWeights_ + (1 - beta2_) * dWeights.array().square().matrix();
        mBiases_ = beta1_ * mBiases_ + (1 - beta1_) * dBias;
        vBiases_ = beta2_ * vBiases_ + (1 - beta2_) * dBias.array().square().matrix();

        MatrixType mHatWeights = mWeights_ / (1 - pow(beta1_, t_));
        MatrixType vHatWeights = vWeights_ / (1 - pow(beta2_, t_));
        VectorType mHatBiases = mBiases_ / (1 - pow(beta1_, t_));
        VectorType vHatBiases = vBiases_ / (1 - pow(beta2_, t_));
        weights.array() -= learningRate_ * mHatWeights.array() / (vHatWeights.array().sqrt() + epsilon_) + learningRate_ * lambda_ * weights.array();
        biases.array() -= learningRate_ * mHatBiases.array() / (vHatBiases.array().sqrt() + epsilon_);
    }

private:
    FloatType beta1_;
    FloatType beta2_;
    FloatType epsilon_;

    int t_;

    MatrixType mWeights_;
    MatrixType vWeights_;
    VectorType mBiases_;
    VectorType vBiases_;
};

}