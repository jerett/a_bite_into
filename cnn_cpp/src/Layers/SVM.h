#include <Eigen/Dense>
#pragma once

using namespace Eigen;

namespace dl {
class SVMLayer : public OutputLayerBase{
public:
    SVMLayer() {}

    // Forward pass of the SVM layer.
    // Sets the output matrix to the input matrix.
    void Forward(const MatrixType& x) override {
        outputMat_ = x;
    }

    // Computes the SVM loss for a single sample.
    FloatType Loss(const VectorType& x, int label) {
        FloatType loss = 0.0;
        int num_classes = x.size();

        FloatType correct_class_score = x(label);
        for (int j = 0; j < num_classes; ++j) {
            if (j != label) {
                FloatType margin = x(j) - correct_class_score + 1; // Delta = 1
                loss += std::max(FloatType(0.0), margin);
            }
        }
        return loss;
    }

    // Computes the SVM loss for the entire batch of predictions.
    FloatType Loss(const MatrixType& pred, const VectorXi& label) override {
        FloatType dataTerm = 0.;
        for (int j = 0; j < pred.cols(); ++j) {
            dataTerm += Loss(pred.col(j), label(j));
        }
        return dataTerm;
    }

    // Backward pass of the SVM layer.
    // Computes the gradients of the SVM loss with respect to the predictions.
    void Backward(const MatrixType& x, const VectorXi& labels) override {
        int num_classes = x.rows();
        int num_samples = x.cols();
        dxMat_ = MatrixType::Zero(num_classes, num_samples);

        for (int i = 0; i < num_samples; ++i) {
            FloatType correct_class_score = x(labels(i), i);
            for (int j = 0; j < num_classes; ++j) {
                if (j == labels(i)) { 
                    continue; 
                }
                FloatType margin = x(j, i) - correct_class_score + 1; // Delta = 1
                if (margin > 0) {
                    dxMat_(j, i) += 1;
                    dxMat_(labels(i), i) -= 1;
                }
            }
        }
    }
};

}