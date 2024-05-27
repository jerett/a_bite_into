#include "ImgTrans.h"

#pragma once

namespace dl {
class Dataset {
public:
	Dataset(const std::shared_ptr<Imgtrans>& imgTrans) :imgTrans_(imgTrans) {}

	virtual int GetLen() = 0;

	virtual bool GetItem(int idx, VectorType& x, int& label) = 0;

	FloatType Process() {
		int len = std::max(GetLen(), 1);
		return startId_ * 100. / len;
	}

	bool NextBatch(int batchSize, MatrixType& batch, VectorXi& labels) {
		if (startId_ >= GetLen() || batchSize >= GetLen()) {
			startId_ = 0;
			return false;
		}
		int endId = std::min(startId_ + batchSize, GetLen());
		startId_ = std::max(startId_, endId - batchSize);

		labels.resize(batchSize);
		VectorType x;
		int label;
		for (int i = startId_; i < endId; ++i) {
			GetItem(i, x, label);
			if (batch.rows() != x.size() || batch.cols() != batchSize) {
				batch.resize(x.size(), batchSize);
			}
			if (imgTrans_) {
				imgTrans_->Apply(x.data());
			}
			batch.col(i - startId_) = x;
			labels(i - startId_) = label;
		}

		startId_ = endId;
		return true;
	}

protected:
	int startId_ = 0;
	std::shared_ptr<Imgtrans> imgTrans_;
};

}