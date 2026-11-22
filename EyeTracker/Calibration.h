#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "Config.h"
#include <chrono>
#include <random>

constexpr int kCalibrationSteps = 5;	//broj koraka nakon "neutralnog"
constexpr int kNeutralDotCount = 5;
constexpr int kDotsPerMovedStep = 3;

bool FitCalibration(const std::vector<cv::Point2f>& features,
				const std::vector<cv::Point2f>& screenPts,
				CalibrationConfig& config);

struct HeadPose 
{ 
	float shiftX
		, shiftY
		, scale
		, yaw		//horizontalni pomeraj nosa u poredjenju sa srednjom tackom
		, pitch		//vertikalni pomeraj nosa u poredjenju sa srednjom tackom
		, roll		//ugao pod kojim je glava
		; 
};

struct CalibrationSample
{
	cv::Point2f feature;	//pogled
	HeadPose pose;			//polozaj glave
	cv::Point2f screen;		//tacka na ekranu
	int dot;				//indeks tacke u koraku
	int step;				//broj koraka (0 = "neutralan")
};

struct FitReport
{
	int used = 0, total = 0;
	double meanErr = 0;         
	double neutralDotErr[kNeutralDotCount] = {};
	double holdOutWith = 0;       
	double holdOutWithout = 0;         
	int holdOutCount = 0;
	bool poseCorrectionOn = false;
};

bool FitCalibration(const std::vector<CalibrationSample>& samples, int stepCount, CalibrationConfig& config, FitReport* report = nullptr);

cv::Point2f MapGazeToScreen(const CalibrationConfig& config, cv::Point2f feature);

class CalibrationSession
{
public:
	void Start(int screenW, int screenH);
	void Cancel();
	bool Active() const { return mActive; }
	double LastError() const { return mReport.meanErr; }

	bool Update(bool hasFeature, cv::Point2f feature, const HeadPose& pose, bool spacePressed, CalibrationConfig& config);
	const FitReport& Report() const { return mReport; }

private:
	enum class Phase { IntroNeutral, WaitMove, Dots };

	cv::Point2f GridPos(int gridIndex) const;
	void ChooseStepDots();
	void ShowDots(int activeIndex);
	void ShowIntro();
	void ShowDotText(bool retry = false);
	void ShowWaitMove(const char* warning);
	void BeginDots();
	bool Capture(bool hasFeature, cv::Point2f feature, const HeadPose& pose,
		cv::Point2f target, int dotId, int settleMs, int sampleMs);

	bool mActive = false;
	Phase mPhase = Phase::IntroNeutral;
	bool mSampling = false;
	int mScreenW = 0, mScreenH = 0;
	int mStep = 0;									//0 = neutral
	int mDotInStep = 0;
	std::vector<int> mStepDots;						//pozicija u gridu (0..8) tacaka u trenutnom koraku
	std::chrono::steady_clock::time_point mPhaseStart;
	std::vector<CalibrationSample> mCurrent;
	std::vector<CalibrationSample> mStepSamples;
	std::vector<CalibrationSample> mAll;
	std::vector<HeadPose> mStepPoses;
	std::mt19937 mRng;
	FitReport mReport;
	bool mCovered[kPoseTerms] = {};
};