#include <Eigen/Dense>
#include "LayerBase.h"
#pragma once

using namespace Eigen;

namespace dl {
/**
 * @brief Max pooling layer class.
 *
 * This class represents a max pooling layer in a neural network.
 * It reduces the spatial dimensions of the input volume by taking the maximum value within each window.
 */
class MaxPool :public LayerBase {
public:
	MaxPool(int inputSize, int inputChannels) {
		inputSize_ = inputSize;
		inputChannels_ = inputChannels;
		outputSize_ = inputSize_ / 2;
	}

	void Forward(const MatrixType& x, bool isTraining) override {
		if (dxMat_.rows() != x.rows() || dxMat_.cols() != x.cols()) {
			dxMat_.resize(x.rows(), x.cols());
		}
		if (outputMat_.rows() != outputSize_ * outputSize_ * inputChannels_ || outputMat_.cols() != x.cols()) {
			outputMat_.resize(outputSize_ * outputSize_ * inputChannels_, x.cols());
		}

		forwardIndices_.clear();
		for (int sampleId = 0; sampleId < x.cols(); ++sampleId) {
			for (int i = 0; i < outputSize_; ++i) {
				for (int j = 0; j < outputSize_; ++j) {
					int iIn = i * 2;
					int jIn = j * 2;
					FloatType* outPtr = outputMat_.col(sampleId).data() + (i * outputSize_ + j) * inputChannels_;
					const FloatType* inPtr1 = x.col(sampleId).data() + (iIn * inputSize_ + jIn) * inputChannels_;
					const FloatType* inPtr2 = x.col(sampleId).data() + (iIn * inputSize_ + jIn + 1) * inputChannels_;
					const FloatType* inPtr3 = x.col(sampleId).data() + ((iIn + 1) * inputSize_ + jIn) * inputChannels_;
					const FloatType* inPtr4 = x.col(sampleId).data() + ((iIn + 1) * inputSize_ + jIn + 1) * inputChannels_;
					for (int n = 0; n < inputChannels_; ++n) {
						FloatType maxv = std::max(inPtr1[n], std::max(inPtr2[n], std::max(inPtr3[n], inPtr4[n])));
						outPtr[n] = maxv;
						if (isTraining) {
							int inId = -1;
							int outId = sampleId * outputMat_.rows() + (i * outputSize_ + j) * inputChannels_ + n;
							if (maxv == inPtr1[n]) {
								inId = sampleId * x.rows() + (iIn * inputSize_ + jIn) * inputChannels_ + n;
							}
							else if (maxv == inPtr2[n]) {
								inId = sampleId * x.rows() + (iIn * inputSize_ + jIn + 1) * inputChannels_ + n;
							}
							else if (maxv == inPtr3[n]) {
								inId = sampleId * x.rows() + ((iIn + 1) * inputSize_ + jIn) * inputChannels_ + n;
							}
							else {
								inId = sampleId * x.rows() + ((iIn + 1) * inputSize_ + jIn + 1) * inputChannels_ + n;
							}

							// Record the mapping between input and output indices during the forward pass.
							// `inId` represents the index of the input element contributing to the maximum value in the output,
							// while `outId` represents the index of the corresponding output element.
							// This mapping is later used during backward propagation to efficiently propagate gradients from the output to the input.
							forwardIndices_.push_back(std::make_tuple(inId, outId));
						}
					}
				}
			}
		}
	}

	// As you see.
	void Backward(const MatrixType& dz) override {
		dxMat_.setZero();

		FloatType* dxPtr = dxMat_.data();
		const FloatType* dzPtr = dz.data();
		for (const auto& iter : forwardIndices_) {
			int inputId, outputId;
			std::tie(inputId, outputId) = iter;
			dxPtr[inputId] = dzPtr[outputId];
		}
	}

	int OutImageSize() override { return outputSize_; }

	int OutChannels() override { return inputChannels_; }

private:
	int inputSize_;
	int inputChannels_;
	int outputSize_;
	// inputId, outputId
	std::vector<std::tuple<int, int>> forwardIndices_;
};
}