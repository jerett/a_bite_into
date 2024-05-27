#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "src/Dataset/Dataset.h"
#include "src/Model.h"
#include "src/Optimizer/AllOptimizers.h"
using namespace Eigen;
using namespace cv;
using namespace dl;

class BinaryDataset :public Dataset{
public:
	BinaryDataset(std::string str, int dataCnt) : Dataset(nullptr) {
		GenerateData(str, dataCnt);
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
	FloatType SinFunc(FloatType x, FloatType y) {
		return 200 * sin(x * 3 * CV_2PI / 600) - y;
	}

	FloatType CircleFunc(FloatType x, FloatType y) {
		auto d1 = fabs(sqrt(x * x + y * y) - 180) - 30;
		auto d2 = fabs(sqrt(x * x + y * y) - 280) - 30;
		auto d3 = fabs(sqrt(x * x + y * y) - 80) - 30;
		return std::min(std::min(d1, d2), d3);
	}

	void GenerateData(std::string str, int dataCnt) {
		data_.clear();
		for (int i = 0; ; ++i) {
			FloatType x = rand() % 601 - 300;
			FloatType y = rand() % 601 - 300;
			FloatType z;
			if (str == "sin") {
				z = SinFunc(x, y);
			}
			if (str == "circle") {
				z = CircleFunc(x, y);
			}
			int label;
			if (z > 10) {
				label = 1;
			}
			else if (z < -10) {
				label = 0;
			}
			else {
				continue;
			}
			std::pair<MatrixType, int> patch;
			patch.first.resize(2, 1);
			patch.first << x, y;
			patch.second = label;
			data_.push_back(patch);
			if (data_.size() == dataCnt) {
				break;
			}
		}
	}

private:
	std::vector<std::pair<MatrixType, int>> data_;
};

class BinaryClassifyModel :public Model {
public:
	BinaryClassifyModel() {
	}

	void SetDataset(std::shared_ptr<BinaryDataset> trainData, std::shared_ptr<BinaryDataset> testData) {
		trainData_ = trainData;
		testData_ = testData;
	}

	void Train() override {
		auto time0 = cv::getTickCount();
		for (int epoch = 1; epoch <= 300; ++epoch) {
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
				
				Backward(output, labels);
			}
			auto time1 = cv::getTickCount();
			std::cout << "\rEpoch: " << epoch << " Accuracy: " << correctPreds * 100. / sum << "   Cost: " << (time1 - time0) / cv::getTickFrequency() << "s          ";
			Test(5);
		}
		Test();
		cv::waitKey();
	}

	void Test(int sampleStep = 1) override {
		auto Draw = [](cv::Mat& img, cv::Point2f pt, cv::Scalar bgr, bool correct) {
			if (correct) {
				cv::circle(img, pt, 3, bgr, -1);
			}
			else {
				cv::Point2f p1 = pt + cv::Point2f(-3, -3);
				cv::Point2f p2 = pt + cv::Point2f(3, -3);
				cv::Point2f p3 = pt + cv::Point2f(-3, 3);
				cv::Point2f p4 = pt + cv::Point2f(3, 3);
				cv::line(img, p1, p4, bgr, 2);
				cv::line(img, p2, p3, bgr, 2);
			}
		};

		cv::Mat show = cv::Mat::zeros(600, 600, CV_8UC3);
		show.setTo(255);
		int correctCnt = 0, sum = 0;
		MatrixType output;
		int radius = (sampleStep + 1) / 2;
		for (int y = 0; y < 600; y+=sampleStep) {
			for (int x = 0; x < 600; x+=sampleStep) {
				MatrixType in;
				in.resize(2, 1);
				in << x - 300, y - 300;
				Forward(in, output, false);
				int predLabel = outputLayerPtr_->Argmax(output);
				if (predLabel == 0) {
					cv::circle(show, cv::Point(x, y), radius, CV_RGB(255, 128, 128), -1);
				}
				else {
					cv::circle(show, cv::Point(x, y), radius, CV_RGB(128, 128, 255), -1);
				}
			}
		}

		MatrixType input;
		VectorXi label;
		int id = 0;
		while (testData_->NextBatch(1, input, label)) {
			Forward(input, output, false);
			int predLabel = outputLayerPtr_->Argmax(output.col(0));
			bool correct = predLabel == label[0] ? 1 : 0;
			correctCnt += correct ? 1 : 0;
			sum++;
			cv::Scalar bgr = label[0] == 0 ? CV_RGB(255, 0, 0) : CV_RGB(0, 0, 255);
			cv::Point2f pt(input(0, 0) + 300, input(1, 0) + 300);
			Draw(show, pt, bgr, correct);
		}

		std::cout << "Test Accuracy: " << FloatType(correctCnt) / sum << "\n";
		cv::imshow("show", show);
		cv::waitKey(1);
	}

private:
	std::shared_ptr<BinaryDataset> trainData_;
	std::shared_ptr<BinaryDataset> testData_;
};

int main() {
	std::shared_ptr<BinaryDataset> trainData = std::make_shared<BinaryDataset>("circle", 1000);
	std::shared_ptr<BinaryDataset> testData = trainData;

	BinaryClassifyModel binModel;
	for (int i = 0; i < 10; ++i) {
		binModel.Dense(64, WithLayer::ReLU);
	}
	binModel.Dense(2);
	binModel.LossFunc("crossEntropyLoss");
	binModel.SetDataset(trainData, testData);
	binModel.SetOptimizer(OptimizerMethod::ADAM, 1e-3, 1e-1);
	binModel.Train();
	binModel.Test();
	return 0;
}