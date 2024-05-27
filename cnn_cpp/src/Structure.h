#include <Eigen/Dense>
using namespace Eigen;

#pragma once

#define _USE_DOUBLE_ 1

#if _USE_DOUBLE_
#define FloatType double
#define MatrixType MatrixXd
#define VectorType VectorXd
#define CV_FLOAT CV_64F
#define CV_FLOAT_C3 CV_64FC3

#else
#define FloatType float
#define MatrixType MatrixXf
#define VectorType VectorXf
#define CV_FLOAT CV_32F
#define CV_FLOAT_C3 CV_32FC3

#endif


