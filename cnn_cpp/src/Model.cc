#include "Model.h"

namespace dl {

Model::Model() {
}

void Model::Dense(int outputSize, uint32_t withLayer) {
	layers_.push_back(std::make_shared<FullConnect>(outputSize));
	canAddConvLayer_ = false;
	if (withLayer & WithLayer::ReLU) {
		Activation();
	}
	if (withLayer & WithLayer::BN) {
		BatchNoralization();
	}
}

void Model::Conv2D(int outChs, int kerSize, int stride, int padding, uint32_t withLayer) {
	Conv2D(outChs, kerSize, prevConvOutWidth_, prevConvOutChannels_, stride, padding, withLayer);
}

void Model::Conv2D(int outChs, int kerSize, int inSize, int inChs, int stride, int padding, uint32_t withLayer) {
	if (!canAddConvLayer_) {
		std::cout << "Error: Cannot add convolution layer here\n";
		return;
	}

	layers_.push_back(std::make_shared<ConvolutionLayer>(inSize, inChs, outChs, kerSize, stride, padding));

	// Update the dimensions for the previous convolution layer
	prevConvOutWidth_ = layers_.back()->OutImageSize();
	prevConvOutChannels_ = layers_.back()->OutChannels();

	if (withLayer & WithLayer::MaxPool) {
		MaxPooling(prevConvOutWidth_, prevConvOutChannels_);
		prevConvOutWidth_ = layers_.back()->OutImageSize();
		prevConvOutChannels_ = layers_.back()->OutChannels();
	}
	if (withLayer & WithLayer::ReLU) {
		Activation();
	}
	if (withLayer & WithLayer::BN) {
		BatchNoralization();
	}
}

void Model::Activation() {
	layers_.push_back(std::make_shared<ReLU>());
}

void Model::MaxPooling(int inputSize, int inputChannels) {
	layers_.push_back(std::make_shared<MaxPool>(inputSize, inputChannels));
}

void Model::BatchNoralization() {
	layers_.push_back(std::make_shared<BatchNorm>());
}

FloatType Model::Loss(const MatrixType& pred, const VectorXi& label, bool useRegularization) {
	FloatType dataTerm = outputLayerPtr_->Loss(pred, label);
	if (useRegularization) {
		FloatType sumL2Norm = 0.;
		FloatType regularizationTerm = 0.;
		for (const auto& ptr : layers_) {
			regularizationTerm += ptr->RegularizationTerm();
		}
		return dataTerm + regularizationTerm;
	}
	return dataTerm;
}

void Model::Forward(const MatrixType& patch, MatrixType& output, bool isTraining) {
	layers_.front()->Forward(patch, isTraining);
	for (int i = 1; i < layers_.size(); ++i) {
		layers_[i]->Forward(layers_[i - 1]->outputMat_, isTraining);
	}

	outputLayerPtr_->Forward(layers_.back()->outputMat_);
	output = outputLayerPtr_->outputMat_;
}

void Model::Backward(const MatrixType& output, const VectorXi& labels) {
	outputLayerPtr_->Backward(output, labels);

	layers_.back()->Backward(outputLayerPtr_->dxMat_);
	for (int i = int(layers_.size()) - 2; i >= 0; --i) {
		layers_[i]->Backward(layers_[i + 1]->dxMat_);
	}
}

//void Model::Train() {
//	auto time0 = cv::getTickCount();
//	int totalIters = iterations_ * (trainData_.size() / batchSize_);
//	int iterCnt = 0;
//	FloatType processCout = 0.;
//	for (int iter = 0; iter < iterations_; ++iter) {
//		FloatType totalLoss = 0.0;
//		int correctPredictions = 0;
//
//		MatrixType input;
//		VectorXi labels;
//		MatrixType output;
//
//		for (int i = 0; i < trainData_.size() - batchSize_ + 1; i += batchSize_) {
//			GetBatch(trainData_, i, i + batchSize_, input, labels);
//
//			Forward(input, output, true);
//
//			totalLoss += Loss(output, labels);
//			for (int j = 0; j < output.cols(); ++j) {
//				correctPredictions += outputLayerPtr_->Argmax(output.col(j)) == labels[j] ? 1 : 0;
//			}
//			Backward(output, labels);
//
//			iterCnt++;
//			FloatType process = iterCnt * 100. / totalIters;
//			auto time1 = cv::getTickCount();
//			FloatType t = (time1 - time0) / cv::getTickFrequency();
//			if (option_.seeProcess) {
//				std::wcout << "\rProcess: " << process << " % Cost: " << t << " s   ";
//				if (process >= processCout + 0.1) {
//					Test(10);
//					processCout = process;
//				}
//			}
//			layers_[0]->ShowDebugInfo();
//			//layers_[6]->ShowDebugInfo();
//		}
//
//		FloatType time1 = cv::getTickCount();
//		FloatType accuracy = static_cast<FloatType>(correctPredictions) / trainData_.size();
//		std::cout << "Iteration " << iter
//			<< ", Average Loss: " << totalLoss / trainData_.size()
//			<< ", Accuracy: " << accuracy
//			<< ", Cost:" << (time1 - time0) / cv::getTickFrequency() << " s\n";
//		Test(1);
//		std::cout << "\n\n";
//
//		layers_[0]->ShowDebugInfo();
//		//layers_[6]->ShowDebugInfo();
//
//		if (option_.adjustLearningRate) {
//			static FloatType lr = learningRate_;
//			lr *= 0.9;
//			lr = std::max(lr, FloatType(0.05 * learningRate_));
//			for (auto& layer : layers_) {
//				layer->SetLearningRateAndLambda(lr, lambda_);
//			}
//		}
//	}
//}
//
//void Model::Test(int sampleStep) {
//	int correctCnt = 0, sum = 0;
//	MatrixType output;
//	for (int i = 0; i < testData_.size(); i += sampleStep) {
//		Forward(testData_[i].first, output, false);
//
//		int predLabel = outputLayerPtr_->Argmax(output);
//		correctCnt += predLabel == testData_[i].second ? 1 : 0;
//		sum++;
//		continue;
//
//		std::cout << "gt, pred, Accuracy: " << testData_[i].second << " " << predLabel << " " << FloatType(correctCnt) / sum << "\n";
//
//		cv::Mat show1d;
//		cv::Mat cvPatch;
//		cv::eigen2cv(testData_[i].first, cvPatch);
//		cvPatch.convertTo(show1d, CV_8U, 255.);
//		cv::Mat show(28, 28, CV_8U, show1d.data);
//		cv::resize(show, show, cv::Size(show.cols * 10, show.rows * 10), cv::INTER_NEAREST);
//		cv::imshow("show", show);
//		cv::waitKey(1);
//	}
//	std::cout << "Test Accuracy: " << FloatType(correctCnt) / sum << "\n";
//}

// Perform gradient checking for debugging.
//void Model::DebugCheckGradients(CheckGradType checkGradType, int layerId, FloatType epsilon) {
//	if (layerId >= layers_.size() || layerId < 0) {
//		std::cout << "Error: layerId = " << layerId << ", out of [" << 0 << ", " << layers_.size() << ")\n";
//		return;
//	}
//
//	for (int iter = 0; iter < iterations_; ++iter) {
//		int batchSize = 10;
//		MatrixType input, output;
//		VectorXi labels;
//
//		for (int i = 0; i < trainData_.size(); i += batchSize) {
//			GetBatch(trainData_, i, i + batchSize, input, labels);
//
//			FloatType positionRate = (rand() % 1000) / 1000.;
//
//			layers_[layerId]->checkGradType_ = checkGradType;
//			bool flag = layers_[layerId]->PerturbForGradientCheck(positionRate, epsilon);
//			Forward(input, output, true);
//			FloatType loss1 = Loss(output, labels, false);
//
//			layers_[layerId]->PerturbForGradientCheck(positionRate, -2 * epsilon);
//			Forward(input, output, true);
//			FloatType loss2 = Loss(output, labels, false);
//
//			layers_[layerId]->PerturbForGradientCheck(positionRate, epsilon);
//			Forward(input, output, true);
//			Backward(output, labels);
//			FloatType backwardGrad = layers_[layerId]->GetGradient(positionRate);
//
//			FloatType numericGrad = (loss1 - loss2) / (2 * epsilon);
//			FloatType error = fabs(backwardGrad - numericGrad);
//			error /= std::max(std::max(fabs(backwardGrad), fabs(numericGrad)), FloatType(1e-20));
//
//			std::cout << "Flag, Numeric grad, Backward grad, Error: "
//				<< flag << " "
//				<< numericGrad << " "
//				<< backwardGrad << " "
//				<< error << "\n";
//			std::getchar();
//		}
//	}
//}
}