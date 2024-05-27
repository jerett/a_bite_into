#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "src/Model.h"
#include "src/Dataset/Dataset.h"
#include "src/Dataset/ImgTrans.h"
#include "src/Optimizer/AllOptimizers.h"

using namespace Eigen;
using namespace dl;

class MNISTDataset : public Dataset {
public:
	MNISTDataset(const std::string& path, bool shuffleFlag, std::shared_ptr<Imgtrans> trans = nullptr) :Dataset(trans) {
		LoadData(path, shuffleFlag);
	}

	int GetLen() override {
		return static_cast<int>(data_.size());
	}

	bool GetItem(int idx, VectorType& x, int& label) override {
		x = data_[idx].first;
		label = data_[idx].second;
		return true;
	}

private:
	bool LoadData(const std::string& path, bool shuffleFlag) {
		auto Seg = [](const cv::Mat& img, int label, std::vector<std::pair<MatrixType, int>>& patches) {
			patches.clear();
			for (int i = 0; i < img.rows; i += 28) {
				for (int j = 0; j < img.cols - 28; j += 28) {
					cv::Rect rect(j, i, 28, 28);
					cv::Mat patch = img(rect).clone();
					cv::Mat patch1d = patch.reshape(0, patch.rows * patch.cols);
					cv::Mat patch1d_f;
					patch1d.convertTo(patch1d_f, CV_FLOAT);
					MatrixType eiPatch;
					cv::cv2eigen(patch1d_f, eiPatch);
					patches.emplace_back(eiPatch, label);
				}
			}
		};

		for (int i = 0; i < 10; ++i) {
			std::stringstream ss;
			ss << path << "/" << i << ".jpg";
			cv::Mat img = cv::imread(ss.str(), 0);
			std::vector<std::pair<MatrixType, int>> data;
			Seg(img, i, data);
			data_.insert(data_.end(), data.begin(), data.end());
		}

		if (shuffleFlag) {
			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(data_.begin(), data_.end(), g);
		}
		return !data_.empty();
	}

private:
	std::vector<std::pair<MatrixType, int>> data_;
};

class CNN : public Model {
public:
	CNN() {
	}

	void SetDataset(std::shared_ptr<MNISTDataset> trainData, std::shared_ptr<MNISTDataset> testData) {
		trainData_ = trainData;
		testData_ = testData;
	}

	void Train() override {
	}

	void Test(int sampleStep = 1) override {
	}

	void DebugCheckGradients(CheckGradType checkGradType, int layerId, FloatType epsilon) {
		if (layerId >= layers_.size() || layerId < 0) {
			std::cout << "Error: layerId = " << layerId << ", out of [" << 0 << ", " << layers_.size() << ")\n";
			return;
		}

		for (int epoch = 1; epoch <= 10; ++epoch) {
			MatrixType batch;
			VectorXi labels;
			MatrixType output;
			FloatType totalLoss = 0.0;
			int correctPreds = 0;
			int iterations = 0;
			int sum = 0;
			while (trainData_->NextBatch(1, batch, labels)) {

				FloatType positionRate = (rand() % 1000) / 1000.;

				layers_[layerId]->checkGradType_ = checkGradType;
				bool flag = layers_[layerId]->PerturbForGradientCheck(positionRate, epsilon);
				Forward(batch, output, true);
				FloatType loss1 = Loss(output, labels, false);

				layers_[layerId]->PerturbForGradientCheck(positionRate, -2 * epsilon);
				Forward(batch, output, true);
				FloatType loss2 = Loss(output, labels, false);

				layers_[layerId]->PerturbForGradientCheck(positionRate, epsilon);
				Forward(batch, output, true);
				Backward(output, labels);
				FloatType backwardGrad = layers_[layerId]->GetGradient(positionRate);

				FloatType numericGrad = (loss1 - loss2) / (2 * epsilon);
				FloatType error = fabs(backwardGrad - numericGrad);
				error /= std::max(std::max(fabs(backwardGrad), fabs(numericGrad)), FloatType(1e-20));

				std::cout << "Flag, Numeric grad, Backward grad, Error: "
					<< flag << " "
					<< numericGrad << " "
					<< backwardGrad << " "
					<< error << "\n";
				std::getchar();
			}
		}
	}

private:
	std::shared_ptr<MNISTDataset> trainData_;
	std::shared_ptr<MNISTDataset> testData_;
};


int main() {
	std::string trainPath = "./data/mnist/train";
	auto trainTrans = Imgtrans::Create(28, 28, 1);
	trainTrans->Normalize(0., 1. / 255);
	auto trainData = std::make_shared<MNISTDataset>(trainPath, true, trainTrans);

	std::string testPath = "./data/mnist/test";
	auto testTrans = Imgtrans::Create(28, 28, 1);
	testTrans->Normalize(0., 1. / 255);
	auto testData = std::make_shared<MNISTDataset>(testPath, false, testTrans);

	CNN model;
	model.Conv2D(16, 3, 28, 1, 1, 1);
	model.Conv2D(16, 3, 1, 1);
	model.Dense(128);
	model.Dense(10, WithLayer::None);
	model.LossFunc("crossEntropyLoss");
	model.SetOptimizer(OptimizerMethod::ADAM, 1e-3, 1e-3);
	model.SetDataset(trainData, testData);

	int layerId = 0;
	FloatType epsilon = sizeof(FloatType) == 64 ? 1e-5 : 1e-3;
	model.DebugCheckGradients(CheckGradType::WEIGHT, layerId, epsilon);
	std::cout << "Done\n";
}