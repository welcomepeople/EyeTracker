#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "Config.h"

// 
bool FitCalibration(const std::vector<cv::Point2f>& features,
				const std::vector<cv::Point2f>& screenPts,
				CalibrationConfig& config);

cv::Point2f MapGazeToScreen(const CalibrationConfig& config, cv::Point2f feature);