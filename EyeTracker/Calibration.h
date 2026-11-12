#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "Config.h"
#include <chrono>

// 
bool FitCalibration(const std::vector<cv::Point2f>& features,
				const std::vector<cv::Point2f>& screenPts,
				CalibrationConfig& config);

cv::Point2f MapGazeToScreen(const CalibrationConfig& config, cv::Point2f feature);

class CalibrationSession
{
public:
	void Start(int screenW, int screenH);
	void Cancel();
	bool Active() const { return mActive; }
	double LastError() const { return mLastError; }

	bool Update(bool hasFeature, cv::Point2f feature, CalibrationConfig& config);

private:
	cv::Point2f DotPos(int i) const;
	void ShowDot();

	bool mActive = false;
	bool mSampling = false;
	int mScreenW = 0, mScreenH = 0;
	int mIndex = 0;
	double mLastError = 0;
	std::chrono::steady_clock::time_point mPhaseStart;
	std::vector<cv::Point2f> mSamples, mFeatures, mScreenPts;
};