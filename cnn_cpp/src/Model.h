#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "Layers/AllLayers.h"

#pragma once

using namespace Eigen;

namespace dl {
// Enum to specify the additional layers to be added after a Convolution or FC layer
namespace WithLayer {
	static uint32_t None = 0;  // No additional layer
	static uint32_t BN = 1;    // Batch Normalization
	static uint32_t ReLU = 1 << 1;    // Rectified Linear Unit
	static uint32_t MaxPool = 1 << 2; // Max Pooling
};


class Model {
public:
	Model();

	// Functions for add layer
	void Dense(int outputSize, uint32_t withLayer = WithLayer::None);
	void Conv2D(int outChs, int kerSize, int stride, int padding, uint32_t withLayer = 0);
	void Conv2D(int outChs, int kerSize, int inSize, int inChs, int stride, int padding, uint32_t withLayer = 0);
	void Activation();
	void MaxPooling(int inputSize, int inputChannels);
	void BatchNoralization();

	void SetOptimizer(OptimizerMethod optimizerMethod, FloatType lr, FloatType lambda) {
		for (auto& layer : layers_) {
			layer->SetOpertimizerMethod(optimizerMethod, lr, lambda);
		}
	}

	void LossFunc(std::string str) {
		if (str == "crossEntropyLoss") {
			outputLayerPtr_ = std::make_shared<Softmax>();
		}
		else if (str == "svmLoss") {
			outputLayerPtr_ = std::make_shared<SVMLayer>();
		}
		else {
			std::cout << "Error: loss function must be one of 'crossEntropyLoss' or 'svmLoss'\n";
		}
	}

	virtual void Train() = 0;

	virtual void Test(int sampleStep = 1) = 0;

	void Forward(const MatrixType& patch, MatrixType& output, bool isTraining);

	FloatType Loss(const MatrixType& pred, const VectorXi& label, bool useRegularization = true);

	//void DebugCheckGradients(CheckGradType checkGradType, int layerId, FloatType epsilon);


protected:
	void Backward(const MatrixType& output, const VectorXi& labels);

public:
	// layers_ stores the various layers of the model, managed using shared_ptr.
	std::vector<std::shared_ptr<LayerBase>> layers_;

	// prevConvOutWidth_ and prevConvOutChannels_ track the width and number of channels 
	// of the previous convolutional layer's output.
	int prevConvOutWidth_ = 0;
	int prevConvOutChannels_ = 0;

protected:
	std::shared_ptr<OutputLayerBase> outputLayerPtr_ = nullptr;
	bool canAddConvLayer_ = true;
};
}