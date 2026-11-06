#include "Calibration.h"
#include <algorithm>

static void PolyTerms(cv::Point2f f, double t[6])
{
	t[0] = 1.0;
	t[1] = f.x;
	t[2] = f.y;
	t[3] = f.x * f.y;
	t[4] = f.x * f.x;
	t[5] = f.y * f.y;
}

bool FitCalibration(const std::vector<cv::Point2f>& features,
					const std::vector<cv::Point2f>& screenPoints,
					CalibrationConfig& config)
{
	size_t n = features.size();
	if (n < 6 || n != screenPoints.size())
		return false;

	cv::Mat A(static_cast<int>(n), 6, CV_64F);
	cv::Mat bx(static_cast<int>(n), 1, CV_64F);
	cv::Mat by(static_cast<int>(n), 1, CV_64F);

	for (int i = 0; i < static_cast<int>(n); i++)
	{
		double t[6];
		PolyTerms(features[i], t);
		for (int j = 0; j < 6; j++)
			A.at<double>(i, j) = t[j];
		bx.at<double>(i) = screenPoints[i].x;
		bx.at<double>(i) = screenPoints[i].y;
	}

	cv::Mat cx, cy;
	if (!cv::solve(A, bx, cx, cv::DECOMP_SVD) || !cv::solve(A, by, cy, cv::DECOMP_SVD))
		return false;

	for (int i = 0; i < 6; i++) 
	{
		config.coeffsX[i] = cx.at<double>(i);
		config.coeffsY[i] = cy.at<double>(i);
	}

	config.valid = true;
	return true;
}

cv::Point2f MapGazeToScreen(const CalibrationConfig& config, cv::Point2f feature)
{
	double t[6];
	PolyTerms(feature, t);

	double sx = 0.0, sy = 0.0;
	for (int i = 0; i < 6; i++)
	{
		sx += config.coeffsX[i] * t[i];
		sy += config.coeffsY[i] * t[i];
	}

	sx = std::clamp(sx, 0.0, static_cast<double>(config.screenW - 1));
	sy = std::clamp(sy, 0.0, static_cast<double>(config.screenH - 1));

	return cv::Point2f(static_cast<float>(sx), static_cast<float>(sy));
}