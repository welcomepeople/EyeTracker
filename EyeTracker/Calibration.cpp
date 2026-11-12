#include "Calibration.h"
#include <algorithm>
#include "Overlay.h"
#include <cstdio>

#include <fstream>

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
		by.at<double>(i) = screenPoints[i].y;
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
	config.calibrated = true;
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

static const int kSettleMs = 1200;
static const int kSampleMs = 2000;
static const size_t kMinSamples = 23;

static cv::Point2f MedianOf(const std::vector<cv::Point2f>& v)
{
	std::vector<float> xs, ys;
	for (const auto& p : v) { xs.push_back(p.x); ys.push_back(p.y); }
	std::nth_element(xs.begin(), xs.begin() + xs.size() / 2, xs.end());
	std::nth_element(ys.begin(), ys.begin() + ys.size() / 2, ys.end());
	return cv::Point2f(xs[xs.size() / 2], ys[ys.size() / 2]);
}

cv::Point2f CalibrationSession::DotPos(int i) const
{
	const float fr[3] = { 0.1f, 0.5f, 0.9f };
	return cv::Point2f(mScreenW * fr[i % 3], mScreenH * fr[i / 3]);
}

void CalibrationSession::ShowDot()
{
	cv::Point2f p = DotPos(mIndex);
	SetCalibrationDot((int)p.x, (int)p.y, true);
	char buf[160];
	sprintf_s(buf, "Calibration %d/9: look at the yellow dot\nKeep your head still. Press Esc to cancel.", mIndex + 1);
	SetOverlayText(buf);
}

void CalibrationSession::Start(int screenW, int screenH) 
{
	mScreenW = screenW;
	mScreenH = screenH;
	mIndex = 0;
	mSampling = false;
	mSamples.clear();
	mFeatures.clear();
	mScreenPts.clear();
	mPhaseStart = std::chrono::steady_clock::now();
	mActive = true;
	ShowDot();
}

void CalibrationSession::Cancel()
{
	mActive = false;
	SetCalibrationDot(0, 0, false);
}

bool CalibrationSession::Update(bool hasFeature, cv::Point2f feature, CalibrationConfig& config)
{
	if (!mActive)
		return false;

	using namespace std::chrono;
	auto now = steady_clock::now();
	auto ms = duration_cast<milliseconds>(now - mPhaseStart).count();

	if (!mSampling)
	{
		if (ms >= kSettleMs)
		{
			mSampling = true;
			mSamples.clear();
			mPhaseStart = now;
		}
		return false;
	}

	if (hasFeature)
		mSamples.push_back(feature);
	if (ms < kSampleMs)
		return false;

	mSampling = false;
	mPhaseStart = now;

	if (mSamples.size() < kMinSamples)
	{
		SetOverlayText("Face not detected - retrying this dot");
		return false;
	}

	mFeatures.push_back(MedianOf(mSamples));
	mScreenPts.push_back(DotPos(mIndex));

	//console.log() type shi
	{
		float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
		for (const auto& p : mSamples)
		{
			minX = (std::min)(minX, p.x); maxX = (std::max)(maxX, p.x);
			minY = (std::min)(minY, p.y); maxY = (std::max)(maxY, p.y);
		}

		cv::Point2f f = mFeatures.back();
		cv::Point2f s = mScreenPts.back();
		std::ofstream log("calibration_log.txt", std::ios::app);
		log << "dot " << mIndex << ": feature=(" << f.x << ", " << f.y << ")"
			<< " x-range=[" << minX << ", " << maxX << "]"
			<< " y-range=[" << minY << ", " << maxY << "]"
			<< " samples=" << mSamples.size()
			<< " screen=(" << s.x << ", " << s.y << ")\n";
	}

	mIndex++;

	if (mIndex < 9)
	{
		ShowDot();
		return false;
	}

	mActive = false;
	SetCalibrationDot(0, 0, false);

	if (!FitCalibration(mFeatures, mScreenPts, config))
	{
		SetOverlayText("Calibration failed - press F9 to retry", 3000);
		return false;
	}

	double err = 0;
	for (size_t i = 0; i < mFeatures.size(); i++)
		err += cv::norm(MapGazeToScreen(config, mFeatures[i]) - mScreenPts[i]);
	mLastError = err / mFeatures.size();
	return true;
}