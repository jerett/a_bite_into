#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "src/Model.h"
#include "src/Dataset/Dataset.h"
#include "src/Dataset/ImgTrans.h"
#include "src/Optimizer/AllOptimizers.h"

using namespace Eigen;
using namespace dl;

class Cifar10Dataset : public Dataset {
public:
	Cifar10Dataset(const std::string& path, bool shuffleFlag, std::shared_ptr<Imgtrans> trans = nullptr) : Dataset(trans) {
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
			for (int i = 0; i < img.rows; i += 32) {
				for (int j = 0; j < img.cols; j += 32) {
					cv::Rect rect(j, i, 32, 32);
					cv::Mat patch = img(rect).clone();
					cv::Mat patch1d = patch.reshape(1, patch.rows * patch.cols * patch.channels());
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
			ss << path << i << ".png";
			cv::Mat img = cv::imread(ss.str());
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

class Cifar10Model : public Model {
public:
	Cifar10Model() {
	}

	void SetDataset(std::shared_ptr<Cifar10Dataset> trainData, std::shared_ptr<Cifar10Dataset> testData) {
		trainData_ = trainData;
		testData_ = testData;
	}

	void Train() override {
		auto time0 = cv::getTickCount();
		for (int epoch = 1; epoch <= 100; ++epoch) {
			MatrixType batch;
			VectorXi labels;
			MatrixType output;
			FloatType totalLoss = 0.0;
			int correctPreds = 0;
			int iterations = 0;
			int sum = 0;
			while (trainData_->NextBatch(64, batch, labels)) {
				Forward(batch, output, true);

				totalLoss += Loss(output, labels);
				for (int j = 0; j < output.cols(); ++j) {
					correctPreds += outputLayerPtr_->Argmax(output.col(j)) == labels[j] ? 1 : 0;
					sum++;
				}
				auto time1 = cv::getTickCount();
				std::cout << "\rEpoch: " << epoch << " Process: " << trainData_->Process() << "% Accuracy: " << correctPreds * 100. / sum << "%  Cost: " << (time1 - time0) / cv::getTickFrequency() << "s          ";

				Backward(output, labels);
			}
			Test();
		}

	}

	void Test(int sampleStep = 1) override {
		int correctCnt = 0, sum = 0;
		MatrixType batch;
		VectorXi label;
		MatrixType output;
		int id = 0;
		while (testData_->NextBatch(1, batch, label)) {
			++id;
			if (id % sampleStep == 0) {
				Forward(batch, output, false);
				int predLabel = outputLayerPtr_->Argmax(output.col(0));
				correctCnt += predLabel == label[0] ? 1 : 0;
				sum++;
			}
		}
		std::cout << "Test Accuracy: " << FloatType(correctCnt) / sum << "\n";
	}

private:
	std::shared_ptr<Cifar10Dataset> trainData_;
	std::shared_ptr<Cifar10Dataset> testData_;
};

int main() {
	std::string trainPath = "./data/cf10/train/";
	auto trainTrans = Imgtrans::Create(32, 32, 3);
	trainTrans->RandomShift(Vector4i(-2, 2, -2, 2));
	trainTrans->RandomResize(Vector2f(0.8f, 1.25f));
	trainTrans->RandomFlip(1);
	trainTrans->Normalize(0.f, 1.f / 255);
	auto trainData = std::make_shared<Cifar10Dataset>(trainPath, true, trainTrans);

	std::string testPath = "./data/cf10/test/";
	auto testTrans = Imgtrans::Create(32, 32, 3);
	testTrans->Normalize(0.f, 1.f / 255);
	auto testData = std::make_shared<Cifar10Dataset>(testPath, false, testTrans);

	auto BN_ReLU_MaxPool = WithLayer::BN | WithLayer::ReLU | WithLayer::MaxPool;
	auto BN_ReLU = WithLayer::BN | WithLayer::ReLU;
	Cifar10Model model;
	model.Conv2D(32, 3, 32, 3, 1, 1, BN_ReLU_MaxPool); // 16*16*32
	model.Conv2D(32, 3, 1, 1, BN_ReLU_MaxPool); // 8*8*32
	model.Conv2D(64, 3, 1, 1, BN_ReLU_MaxPool); // 4*4*64
	model.Conv2D(128, 3, 1, 1, BN_ReLU); // 4*4*128
	model.Dense(128, BN_ReLU);
	model.Dense(10, WithLayer::None);
	model.LossFunc("crossEntropyLoss");
	model.SetOptimizer(OptimizerMethod::ADAM, 1e-3f, 1e-2f);
	model.SetDataset(trainData, testData);

	std::cout << "Training...\n";
	model.Train();
	std::cout << "\nTesting...\n";
	model.Test();
	std::cout << "Done\n";
	std::getchar();
}