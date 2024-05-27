#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "Structure.h"

#pragma once

using namespace Eigen;
using namespace cv;

namespace dl {
class Imgtrans {
public:
	static std::shared_ptr<Imgtrans> Create(int w, int h, int ch) {
		return std::make_shared<Imgtrans>(w, h, ch);
	}

	using TransformFunc = std::function<void(FloatType*)>;
	Imgtrans(int w, int h, int ch) :width_(w), height_(h), channels_(ch) {}

	void Apply(FloatType* data) {
		for (auto& t : transforms_) {
			t(data);
		}
	}

	void RandomShift(Vector4i range) {
		Add([this, range](FloatType* data) {this->RandomShiftImpl(data, range); });
	}

	void RandomResize(Vector2f scaleRange) {
		Add([this, scaleRange](FloatType* data) {this->RandomResizeImpl(data, scaleRange); });
	}

	void RandomFlip(int axis) {
		Add([this, axis](FloatType* data) {this->RandomFlipImpl(data, axis); });
	}

	void Normalize(FloatType mean, FloatType scale) {
		Add([this, mean, scale](FloatType* data) {this->NormalizeImpl(data, mean, scale); });
	}

private:
	void Add(TransformFunc func) {
		transforms_.push_back(func);
	}

	void RandomShiftImpl(FloatType* data, Vector4i range) {
		int minx = range[0];
		int maxx = range[1];
		int miny = range[2];
		int maxy = range[3];
		assert(maxx > minx && maxy > miny);
		int shiftx = rand() % (maxx - minx + 1) + minx;
		int shifty = rand() % (maxy - miny + 1) + miny;
		cv::Mat img(height_, width_, CV_MAKETYPE(CV_FLOAT, channels_), data);
		cv::Mat img2 = img.clone();
		int fromx = abs(shiftx), tox = 0;
		int fromy = abs(shifty), toy = 0;
		if (shiftx > 0) {
			std::swap(fromx, tox);
		}
		if (shifty > 0) {
			std::swap(fromy, toy);
		}

		int rectw = width_ - abs(shiftx);
		int recth = height_ - abs(shifty);
		img(cv::Rect(fromx, fromy, rectw, recth)).copyTo(img2(cv::Rect(tox, toy, rectw, recth)));
		std::memcpy(data, img2.data, sizeof(FloatType) * width_ * height_ * channels_);
	}

	void RandomResizeImpl(FloatType* data, Vector2f scaleRange) {
		float lower = scaleRange[0];
		float upper = scaleRange[1];
		assert(upper >= lower && lower > 0);
		float scale = (rand() % 10001) / 10000.f;
		scale = scale * (upper - lower) + lower;
		cv::Mat img(height_, width_, CV_MAKETYPE(CV_FLOAT, channels_), data);
		cv::Mat tmp;
		cv::Size size(width_ * scale, height_ * scale);
		cv::resize(img, tmp, size);
		cv::Rect fromRect, toRect;
		if (scale > 1) {
			fromRect = cv::Rect((tmp.cols - width_ + 1) / 2, (tmp.rows - height_ + 1) / 2, width_, height_);
			toRect = cv::Rect(0, 0, width_, height_);
		}
		else {
			fromRect = cv::Rect(0, 0, tmp.cols, tmp.rows);
			toRect = cv::Rect((width_ - tmp.cols + 1) / 2, (height_ - tmp.rows + 1) / 2, tmp.cols, tmp.rows);
		}
		cv::Mat img2 = img.clone();
		tmp(fromRect).copyTo(img2(toRect));
		std::memcpy(data, img2.data, sizeof(FloatType) * width_ * height_ * channels_);
	}

	void RandomFlipImpl(FloatType* data, int axis) {
		if (rand() % 2 == 0) {
			cv::Mat img(height_, width_, CV_MAKETYPE(CV_FLOAT, channels_), data);
			cv::Mat img2;
			cv::flip(img, img2, axis);
			std::memcpy(data, img2.data, sizeof(FloatType) * width_ * height_ * channels_);
		}
	}

	void NormalizeImpl(FloatType* data, FloatType mean, FloatType scale) {
		assert(scale > 0);
		for (int i = 0; i < width_ * height_ * channels_; ++i) {
			data[i] = (data[i] - mean) * scale;
		}
	}

private:
	std::vector<TransformFunc> transforms_;
	int width_;
	int height_;
	int channels_;
};

}