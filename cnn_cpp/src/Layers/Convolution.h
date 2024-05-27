#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "LayerBase.h"

#pragma once

using namespace Eigen;

namespace dl {
/**
* @brief Convolutional layer for deep learning models.
*
* The ConvolutionLayer class implements a convolutional layer, which applies convolution operations
* to input data using learnable filters (kernels). It supports forward and backward propagation
* as well as visualization of learned filters for debugging purposes.
*/
class ConvolutionLayer :public LayerBase {
public:
	ConvolutionLayer(int inSize, int inChs, int outChs, int kerSize, int stride, int padding) : 
		inputSize_(inSize), inputChannels_(inChs), outputChannels_(outChs), kernelSize_(kerSize), stride_(stride), padding_ (padding){

		// Compute output size based on input size, kernel size, stride, and padding
		outputSize_ = (inputSize_ + 2 * padding_ - kernelSize_) / stride_ + 1;

		// He. Initialization
		FloatType stddev = sqrt(2.0 / (inputChannels_ * kernelSize_ * kernelSize_));
		W_ = MatrixType::Random(inputChannels_ * kernelSize_ * kernelSize_, outputChannels_) * stddev;

		dW_ = MatrixType::Zero(inputChannels_ * kernelSize_ * kernelSize_, outputChannels_);
		biases_ = VectorType::Zero(outputChannels_);
		flatanSize_ = outChs * outputSize_ * outputSize_;

		InitForwardIndices();
	}

	void Forward(const MatrixType& inputMat, bool isTraining) override {
		int samples = inputMat.cols();
		inputData_ = inputMat.data();
		inputRows_ = inputMat.rows();
		outputMat_.resize(flatanSize_, samples);
		outputMat_.setZero();

		const auto wData = W_.data();
		for (int sampleId = 0; sampleId < inputMat.cols(); ++sampleId) {
			const auto outPtr = outputMat_.data() + sampleId * outputMat_.rows();
			const auto inputColPtr = inputData_ + sampleId * inputRows_;
			for (const auto& iter : forwardIndices_) {
				int kerId, inputId, outputId;
				std::tie(kerId, inputId, outputId) = iter;
				const auto kerPtr = wData + kerId;
				const auto inputPtr = inputColPtr + inputId;
#if _USE_DOUBLE_
				Map<VectorType> v1(kerPtr, inputChannels_);
				Map<const VectorType> v2(inputPtr, inputChannels_);
				FloatType sum = v1.dot(v2);
				outPtr[outputId] += sum;
#else
				{
					float sum = 0.f;
					__m128 sum_vec = _mm_setzero_ps();

					for (int n = 0; n + 3 < inputChannels_; n += 4) {
						__m128 ker = _mm_loadu_ps(kerPtr + n);
						__m128 inp = _mm_loadu_ps(inputPtr + n);
						__m128 prod = _mm_mul_ps(ker, inp);
						sum_vec = _mm_add_ps(sum_vec, prod);
					}

					float temp[4];
					_mm_storeu_ps(temp, sum_vec);
					sum = temp[0] + temp[1] + temp[2] + temp[3];

					for (int n = inputChannels_ - (inputChannels_ % 4); n < inputChannels_; ++n) {
						sum += kerPtr[n] * inputPtr[n];
					}
					outPtr[outputId] += sum;
				}
#endif
			}
		}

		int totalSize = outputMat_.rows() * outputMat_.cols();
		auto outPtr = outputMat_.data();
		for (int i = 0; i < totalSize; i += outputChannels_) {
			for (int n = 0; n < outputChannels_; ++n) {
				outPtr[i + n] += biases_[n];
			}
		}
	}

	void Backward(const MatrixType& dzMat) override {
		auto t0 = cv::getTickCount();
		int samples = dzMat.cols();
		dW_.setZero();
		if (dxMat_.rows() != inputRows_ || dxMat_.cols() != dzMat.cols()) {
			dxMat_.resize(inputRows_, dzMat.cols());
		}
		dxMat_.setZero();
		dBiases_.resize(outputChannels_);
		dBiases_.setZero();
		
		// Calculate gradients for biases
		int totalSize = dzMat.rows() * dzMat.cols();
		auto dzData = dzMat.data();
		for (int i = 0; i < totalSize; i += outputChannels_) {
			Map<const VectorType> v(dzData + i, outputChannels_);
			dBiases_.noalias() += v;
		}
		auto t1 = cv::getTickCount();
		// Calculate gradients for weights and input data
		FloatType* wData = W_.data();
		FloatType* dwData = dW_.data();
		for (int sampleId = 0; sampleId < dzMat.cols(); ++sampleId) {
			const auto dzPtr = dzMat.data() + sampleId * dzMat.rows();
			const auto inputColPtr = inputData_ + sampleId * inputRows_;
			auto dxColPtr = dxMat_.data() + sampleId * inputRows_;
			for (const auto& iter : forwardIndices_) {
				int kerId, inputId, outputId;
				std::tie(kerId, inputId, outputId) = iter;
				const auto dout = dzPtr[outputId];
				const auto kerPtr = wData + kerId;
				const auto inputPtr = inputColPtr + inputId;
				auto dxPtr = dxColPtr + inputId;
				auto dwPtr = dwData + kerId;
#if _USE_DOUBLE_
				for (int n = 0; n < inputChannels_; ++n) {
					dwPtr[n] += dout * inputPtr[n];
					dxPtr[n] += dout * kerPtr[n];
				}
#else
				for (int n = 0; n <= inputChannels_ - 4; n += 4) {
					__m128 dout_vec = _mm_set1_ps(dout); 
					__m128 inp_vec = _mm_loadu_ps(inputPtr + n);
					__m128 ker_vec = _mm_loadu_ps(kerPtr + n);

					__m128 dw_vec = _mm_loadu_ps(dwPtr + n);
					__m128 dx_vec = _mm_loadu_ps(dxPtr + n);

					__m128 dw_update = _mm_mul_ps(dout_vec, inp_vec);
					__m128 dx_update = _mm_mul_ps(dout_vec, ker_vec);

					dw_vec = _mm_add_ps(dw_vec, dw_update);
					dx_vec = _mm_add_ps(dx_vec, dx_update);

					_mm_storeu_ps(dwPtr + n, dw_vec);
					_mm_storeu_ps(dxPtr + n, dx_vec);
				}
				for (int n = inputChannels_ - (inputChannels_ % 4); n < inputChannels_; ++n) {
					dwPtr[n] += dout * inputPtr[n];
					dxPtr[n] += dout * kerPtr[n];
				}
#endif
			}
		}
		auto t2 = cv::getTickCount();

		dwUpdate_ = dW_ / samples;
		dbUpdate_ = dBiases_ / samples;
		optimizer_->Update(W_, dwUpdate_, biases_, dbUpdate_);

		auto t3 = cv::getTickCount();
		double fms = 1e3 / cv::getTickFrequency();
		//std::cout << "backward: " << (t1 - t0) * fms << " " << (t2-t1)*fms << " " << (t3-t2)*fms << "\n";
	}

	int OutImageSize() override { return outputSize_; }

	int OutChannels() override { return outputChannels_; }

	void ShowDebugInfo() override {
		int n = int(sqrt(outputChannels_) + 1);
		cv::Mat show = cv::Mat::zeros(n * (outputSize_+1), n * (outputSize_+1), CV_FLOAT);
		for (int sample = 0; sample < outputMat_.cols(); ++sample) {
			auto data = outputMat_.col(sample).data();
			for (int channel = 0; channel < outputChannels_; ++channel) {
				cv::Mat local(outputSize_, outputSize_, CV_FLOAT);
				for (int i = 0; i < outputSize_; ++i) {
					for (int j = 0; j < outputSize_; ++j) {
						local.at<FloatType>(i, j) = data[(i * outputSize_ + j) * outputChannels_ + channel];
					}
				}
				int y = (channel / n) * (outputSize_+1);
				int x = (channel % n) * (outputSize_+1);
				local.copyTo(show(cv::Rect(y, x, outputSize_, outputSize_)));
			}
		}


		cv::Mat showKer = cv::Mat::zeros(n * kernelSize_, n * kernelSize_, CV_32FC3);
		if (inputChannels_ == 3) {
			for (int ch = 0; ch < outputChannels_; ++ch) {
				auto kerData = W_.data() + ch * kernelSize_ * kernelSize_ * inputChannels_;
				cv::Mat local(kernelSize_, kernelSize_, CV_32FC3, kerData);
				int y = (ch / n) * kernelSize_;
				int x = (ch % n) * kernelSize_;
				local.copyTo(showKer(cv::Rect(x, y, kernelSize_, kernelSize_)));
			}
		}
	}

	virtual FloatType RegularizationTerm() override {
		if (W_.rows() > 0 && W_.cols() > 0) {
			return W_.norm() * 0.5 * optimizer_->Lambda();
		}
		return 0.;
	}

private:
	void InitForwardIndices() {
		forwardIndices_.clear();
		int outWidth = outputSize_;
		int outHeight = outputSize_;
		int halfKSize = kernelSize_ / 2;
		int inWidth = inputSize_;
		int inHeight = inputSize_;
		for (int outChannel = 0; outChannel < outputChannels_; ++outChannel) {
			int tmpKernel = outChannel * W_.rows();
			for (int i = 0; i < outHeight; ++i) {
				for (int j = 0; j < outWidth; ++j) {
					int cy = i * stride_;
					int cx = j * stride_;
					int kernelId = -1;
					int outputId = (i * outWidth + j) * outputChannels_ + outChannel;
					for (int y = cy - halfKSize; y <= cy + halfKSize; ++y) {
						for (int x = cx - halfKSize; x <= cx + halfKSize; ++x) {
							++kernelId;
							// Skip padding if outside the input image boundaries
							if (x < 0 || x >= inWidth || y < 0 || y >= inHeight) {
								continue;
							}
							int kerPosition = kernelId * inputChannels_;
							int inputPosition = (y * inWidth + x) * inputChannels_;
							int kernelIdStart = tmpKernel + kerPosition;
							forwardIndices_.push_back(std::make_tuple(kernelIdStart, inputPosition, outputId));
						}
					}
				}
			}
		}
	}

public:
	int inputSize_;
	int inputChannels_;
	int outputChannels_; // 每列一个卷积核，共outputChannels_个卷积核（列）. 
	int kernelSize_;
	int stride_;
	int padding_;
	int outputSize_;
	const FloatType* inputData_;
	int inputRows_;
	int flatanSize_;
	MatrixType dwUpdate_;
	VectorType dbUpdate_;
	// kernelId, inputId, outputId
	std::vector<std::tuple<int, int, int>> forwardIndices_;
	VectorType restoreIn_;
	VectorType restoreKer_;
};
}
