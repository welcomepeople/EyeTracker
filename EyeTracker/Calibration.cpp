#include "Calibration.h"
#include "Overlay.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>

//---------------------- podesavanja
//static const double kMinStdDev[kPoseTerms] = { 0.05, 0.05, 3.0, 0.03, 0.03 };	//koliko sme da varira koeficijent
static const double kMinStdDev[kPoseTerms] = { 0.05, 0.05, 3.0, 0.01, 0.01 };	//koliko sme da varira koeficijent
static const double kMinPoseChange[kPoseTerms] = { 0.2, 0.2, 6.0, 0.05, 0.05 };
static const double kCoverageThreshold[kPoseTerms] = { 0.3, 0.15, 8.0, 0.06, 0.08 };
static const double kRidge = 0.02;

static const char* kPoseTermNames[kPoseTerms] =
{
	"pomeri se na stranu",
	"pomeri se gore ili dole",
	"pomeri se blize ili dalje",
	"okreni glavu levo ili desno",
	"nagni glavu napred ili nazad"
};

static const int kSettleMs = 1000;
static const int kSampleMs = 2500;
static const size_t kMinSamples = 30;

static const int kNeutralGridDots[kNeutralDotCount] = { 0, 2, 8, 6, 4 };

//---------------------- helper
static bool Fail(const char* reason)
{
	std::ofstream log("calibration_log.txt", std::ios::app);
	log << "FitCalibration failed: " << reason << "\n";
	return false;
}

static void PolyTerms(cv::Point2f f, double t[6])
{
	t[0] = 1.0;
	t[1] = f.x;
	t[2] = f.y;
	t[3] = f.x * f.y;
	t[4] = f.x * f.x;
	t[5] = f.y * f.y;
}

static void PoseToArray(const HeadPose& p, double out[kPoseTerms])
{
	out[0] = p.shiftX;
	out[1] = p.shiftY;
	out[2] = p.scale;
	out[3] = p.yaw;
	out[4] = p.pitch;
}

static bool IsFiniteSample(const CalibrationSample& sample)
{
	double poseValues[kPoseTerms];
	PoseToArray(sample.pose, poseValues);

	bool isFinite = std::isfinite(sample.feature.x) && std::isfinite(sample.feature.y);
	for (int i = 0; i < kPoseTerms; i++)
		isFinite = isFinite && std::isfinite(poseValues[i]);
	return isFinite;
}

static HeadPose MeanPose(const std::vector<CalibrationSample>& samples)
{
	HeadPose mean{};
	if (samples.empty())
		return mean;

	double sums[kPoseTerms] = {};
	for (size_t i = 0; i < samples.size(); i++)
	{
		double poseValues[kPoseTerms];
		PoseToArray(samples[i].pose, poseValues);
		for (int j = 0; j < kPoseTerms; j++)
			sums[j] += poseValues[j];
	}

	const double count = static_cast<double>(samples.size());
	mean.shiftX = static_cast<float>(sums[0] / count);
	mean.shiftY = static_cast<float>(sums[1] / count);
	mean.scale = static_cast<float>(sums[2] / count);
	mean.yaw = static_cast<float>(sums[3] / count);
	mean.pitch = static_cast<float>(sums[4] / count);
	return mean;
}

static bool PoseFarEnough(const HeadPose& pose, const std::vector<HeadPose>& stepPoses)
{
	double current[kPoseTerms];
	PoseToArray(pose, current);

	for (size_t i = 0; i < stepPoses.size(); i++)
	{
		double earlier[kPoseTerms];
		PoseToArray(stepPoses[i], earlier);

		double largestChange = 0;
		for (int j = 0; j < kPoseTerms; j++)
			largestChange = (std::max)(largestChange, std::abs(current[j] - earlier[j]) / kMinPoseChange[j]);
		
		if (largestChange < 1.0)
			return false;
	}
	return true;
}

static bool AllCovered(const bool covered[kPoseTerms])
{
	for (int i = 0; i < kPoseTerms; i++)
		if (!covered[i])
			return false;
	return true;
}

static std::string MissingTermsText(const bool covered[kPoseTerms])
{
	std::string text;
	for (int i = 0; i < kPoseTerms; i++)
	{
		if (covered[i])
			continue;
		if (!text.empty())
			text += ", ";
		text += kPoseTermNames[i];
	}
	return text;
}

static void UpdateCoverage(const HeadPose& neutralPose, const HeadPose& stepPose, bool covered[kPoseTerms])
{
	double neutral[kPoseTerms], step[kPoseTerms];
	PoseToArray(neutralPose, neutral);
	PoseToArray(stepPose, step);

	for (int i = 0; i < kPoseTerms; i++)
		if (std::abs(step[i] - neutral[i]) >= kCoverageThreshold[i])
			covered[i] = true;
}

static bool MovesMissingTerm(const HeadPose& pose, const HeadPose& neutralPose, const bool covered[kPoseTerms])
{
	double current[kPoseTerms], neutral[kPoseTerms];
	PoseToArray(pose, current);
	PoseToArray(neutralPose, neutral);

	for (int i = 0; i < kPoseTerms; i++)
		if (!covered[i] && std::abs(current[i] - neutral[i]) >= kCoverageThreshold[i])
			return true;
	return false;
}

static bool SolvePlane(const std::vector<CalibrationSample>& samples
						, const std::vector<int>& sampleIndices
						, cv::Mat& coeffsX
						, cv::Mat& coeffsY)
{
	const int sampleCount = static_cast<int>(sampleIndices.size());

	cv::Mat A(sampleCount, 3, CV_64F);
	cv::Mat targetX(sampleCount, 1, CV_64F);
	cv::Mat targetY(sampleCount, 1, CV_64F);

	for (int i = 0; i < sampleCount; i++)
	{
		const CalibrationSample& sample = samples[sampleIndices[i]];
		A.at<double>(i, 0) = 1.0;
		A.at<double>(i, 1) = sample.feature.x;
		A.at<double>(i, 2) = sample.feature.y;
		targetX.at<double>(i) = sample.screen.x;
		targetY.at<double>(i) = sample.screen.y;
	}

	return cv::solve(A, targetX, coeffsX, cv::DECOMP_SVD)
		&& cv::solve(A, targetY, coeffsY, cv::DECOMP_SVD);
}

static bool SolveRidge(const std::vector<CalibrationSample>& samples
					, const std::vector<int>& idx
					, const double poseRef[kPoseTerms]
					, const double poseStdDev[kPoseTerms]
					, cv::Mat& coeffsX
					, cv::Mat& coeffsY)
{
	const int sampleCount = static_cast<int>(idx.size());
	const int columnCount = 3 + kPoseTerms;

	cv::Mat A(sampleCount, columnCount, CV_64F);
	cv::Mat targetX(sampleCount, 1, CV_64F);
	cv::Mat targetY(sampleCount, 1, CV_64F);

	for (int i = 0; i < sampleCount; i++)
	{
		const CalibrationSample& sample = samples[idx[i]];
		
		double poseValues[kPoseTerms];
		PoseToArray(sample.pose, poseValues);

		A.at<double>(i, 0) = 1.0;
		A.at<double>(i, 1) = sample.feature.x;
		A.at<double>(i, 2) = sample.feature.y;
		for (int j = 0; j < kPoseTerms; j++)
		{
			A.at<double>(i, 3 + j) = (poseValues[j] - poseRef[j]) / poseStdDev[j];
		}

		targetX.at<double>(i) = sample.screen.x;
		targetY.at<double>(i) = sample.screen.y;
	}

	cv::Mat normalMat = A.t() * A;
	for (int i = 3; i < columnCount; i++)
	{
		normalMat.at<double>(i, i) += kRidge * sampleCount;
	}

	cv::Mat rhsX = A.t() * targetX;
	cv::Mat rhsY = A.t() * targetY;
	return cv::solve(normalMat, rhsX, coeffsX, cv::DECOMP_SVD) 
		&& cv::solve(normalMat, rhsY, coeffsY, cv::DECOMP_SVD);
}

static cv::Point2f Predict(const cv::Mat& coeffsX
							, const cv::Mat& coeffsY
							, const double poseRef[kPoseTerms]
							, const double poseStdDev[kPoseTerms]
							, const CalibrationSample& sample
							, bool usePose)
{
	double x = coeffsX.at<double>(0) + coeffsX.at<double>(1) * sample.feature.x
									+ coeffsX.at<double>(2) * sample.feature.y;
	double y = coeffsY.at<double>(0) + coeffsY.at<double>(1) * sample.feature.x
									+ coeffsY.at<double>(2) * sample.feature.y;

	if (usePose)
	{
		double poseValues[kPoseTerms];
		PoseToArray(sample.pose, poseValues);

		for (int i = 0; i < kPoseTerms; i++)
		{
			double standardized = (poseValues[i] - poseRef[i]) / poseStdDev[i];
			x += coeffsX.at<double>(3 + i) * standardized;
			y += coeffsY.at<double>(3 + i) * standardized;
		}
	}
	return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
}

//---------------------- 
bool FitCalibration(const std::vector<CalibrationSample>& samples
	, int stepCount
	, CalibrationConfig& config
	, FitReport* report)
{
	//{ 
	//	std::ofstream log("calibration_log.txt", std::ios::app); log << "FitCalibration: " << samples.size() << " samples in\n";
	//}
	if (report) *report = FitReport();

	std::vector<int> finiteIndices;
	for (int i = 0; i < static_cast<int>(samples.size()); i++)
		if (IsFiniteSample(samples[i]))
			finiteIndices.push_back(i);
	const int totalSamples = static_cast<int>(finiteIndices.size());

	#pragma region Prosek i Standardna Devijacija
	double poseRef[kPoseTerms] = {};
	int neutralCount = 0;
	for (int i = 0; i < totalSamples; i++)
	{
		const CalibrationSample& sample = samples[finiteIndices[i]];
		if (sample.step != 0)
			continue;
		double poseValues[kPoseTerms];
		PoseToArray(sample.pose, poseValues);
		for (int j = 0; j < kPoseTerms; j++)
			poseRef[j] += poseValues[j];
		neutralCount++;
	}
	if (neutralCount < 60)
		return Fail("< 60 usable neutral samples");
	for (int j = 0; j < kPoseTerms; j++)
		poseRef[j] /= neutralCount;

	double poseStdDev[kPoseTerms] = {};
	for (int i = 0; i < totalSamples; i++)
	{
		double poseValues[kPoseTerms];
		PoseToArray(samples[finiteIndices[i]].pose, poseValues);
		for (int j = 0; j < kPoseTerms; j++)
		{
			double difference = poseValues[j] - poseRef[j];
			poseStdDev[j] += difference * difference;
		}
	}

	for (int i = 0; i < kPoseTerms; i++)
		poseStdDev[i] = (std::max)(std::sqrt(poseStdDev[i] / totalSamples), kMinStdDev[i]);
	#pragma endregion

#pragma region AAAAAAAAAAAAAAAAAA
	cv::Mat coeffsX, coeffsY;
	std::vector<int> usedIndices = finiteIndices;
	for (int i = 0; i < 2; i++)
	{
		if (usedIndices.size() < 100)
			return Fail("< 100 samples after trimming");
		if(!SolveRidge(samples, usedIndices, poseRef, poseStdDev, coeffsX, coeffsY))
			return Fail("solver failed");

		if (i == 1)
			break;

		std::vector<double> errors;
		for (size_t j = 0; j < usedIndices.size(); j++)
		{
			const CalibrationSample& sample = samples[usedIndices[j]];
			errors.push_back(cv::norm(Predict(coeffsX, coeffsY, poseRef, poseStdDev, sample, true) - sample.screen));
		}

		std::vector<double> sortedErrors = errors;
		std::sort(sortedErrors.begin(), sortedErrors.end());
		double errorLimit = (std::max)(3.0 * sortedErrors[sortedErrors.size() / 2], 100.0);
	
		std::vector<int> keptIndices;
		for (size_t j = 0; j < usedIndices.size(); j++)
			if (errors[j] <= errorLimit)
				keptIndices.push_back(usedIndices[j]);
		if (keptIndices.size() == usedIndices.size())
			break;
		usedIndices = keptIndices;
	}
#pragma endregion

	std::vector<int> neutralIndices;
	for (size_t i = 0; i < usedIndices.size(); i++)
		if (samples[usedIndices[i]].step == 0)
			neutralIndices.push_back(usedIndices[i]);

	cv::Mat neutralX, neutralY;
	if (neutralIndices.size() < 30 || !SolvePlane(samples, neutralIndices, neutralX, neutralY))
		return Fail("neutral plane failed");

	double holdWithSum = 0, holdWithoutSum = 0;
	int holdCount = 0;
	for (int i = 1; i < stepCount; i++)
	{
		std::vector<int> trainIndices, testIndices;
		for (size_t j = 0; j < usedIndices.size(); j++)
		{
			if (samples[usedIndices[j]].step == i)
				testIndices.push_back(usedIndices[j]);
			else
				trainIndices.push_back(usedIndices[j]);
		}
		if (testIndices.empty() || trainIndices.size() < 100)
			continue;

		cv::Mat heldX, heldY;
		if (!SolveRidge(samples, trainIndices, poseRef, poseStdDev, heldX, heldY))
			continue;

		for (size_t j = 0; j < testIndices.size(); j++)
		{
			const CalibrationSample& sample = samples[testIndices[j]];
			holdWithSum += cv::norm(Predict(heldX, heldY, poseRef, poseStdDev, sample, true) - sample.screen);
			holdWithoutSum += cv::norm(Predict(neutralX, neutralY, poseRef, poseStdDev, sample, false) - sample.screen);
			holdCount++;
		}
	}
	const double holdWith = holdCount > 0 ? holdWithSum / holdCount : 0.0;
	const double holdWithout = holdCount > 0 ? holdWithoutSum / holdCount : 0.0;
	const bool useCorrection = holdCount > 0 && holdWith < 0.9 * holdWithout;

#pragma region Cuvanje rezultata
	const cv::Mat& finalX = useCorrection ? coeffsX : neutralX;
	const cv::Mat& finalY = useCorrection ? coeffsY : neutralY;

	for (int i = 0; i < 6; i++)
	{
		config.coeffsX[i] = (i < 3) ? finalX.at<double>(i) : 0.0;
		config.coeffsY[i] = (i < 3) ? finalY.at<double>(i) : 0.0;
	}
	for (int i = 0; i < kPoseTerms; i++)
	{
		config.poseRef[i] = poseRef[i];
		config.poseX[i] = useCorrection ? coeffsX.at<double>(3 + i) / poseStdDev[i] : 0.0;
		config.poseY[i] = useCorrection ? coeffsY.at<double>(3 + i) / poseStdDev[i] : 0.0;
		if (!std::isfinite(config.poseX[i]) || !std::isfinite(config.poseY[i]))
			return Fail("pose coefficient is not finite");
	}
	for (int i = 0; i < 3; i++)
		if (!std::isfinite(config.coeffsX[i]) || !std::isfinite(config.coeffsY[i]))
			return Fail("plane coefficient is not finite");

	config.hasPose = useCorrection;
	config.valid = true;
	config.calibrated = true;

	if (report)
	{
		report->total = totalSamples;
		report->used = static_cast<int>(usedIndices.size());

		double errorSum = 0;
		for (size_t i = 0; i < usedIndices.size(); i++)
		{
			const CalibrationSample& sample = samples[usedIndices[i]];
			errorSum += cv::norm(Predict(finalX, finalY, poseRef, poseStdDev, sample, useCorrection) - sample.screen);
		}
		report->meanErr = errorSum / usedIndices.size();

		int dotSampleCount[kNeutralDotCount] = {};
		for (int i = 0; i < totalSamples; i++)
		{
			const CalibrationSample& sample = samples[finiteIndices[i]];
			if (sample.step != 0 || sample.dot < 0 || sample.dot >= kNeutralDotCount)
				continue;
			report->neutralDotErr[sample.dot] +=
				cv::norm(Predict(finalX, finalY, poseRef, poseStdDev, sample, useCorrection) - sample.screen);
			++dotSampleCount[sample.dot];
		}
		for (int i = 0; i < kNeutralDotCount; i++)
			if (dotSampleCount[i] > 0)
				report->neutralDotErr[i] /= dotSampleCount[i];

		report->holdOutWith = holdWith;
		report->holdOutWithout = holdWithout;
		report->holdOutCount = holdCount;
		report->poseCorrectionOn = useCorrection;
	}
#pragma endregion
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

	if (!std::isfinite(sx) || !std::isfinite(sy))
		return cv::Point2f(config.screenW * 0.5f, config.screenH * 0.5f);

	sx = std::clamp(sx, 0.0, static_cast<double>(config.screenW - 1));
	sy = std::clamp(sy, 0.0, static_cast<double>(config.screenH - 1));

	return cv::Point2f(static_cast<float>(sx), static_cast<float>(sy));
}

//---------------------- calibration

cv::Point2f CalibrationSession::GridPos(int i) const
{
	const float fr[3] = { 0.1f, 0.5f, 0.9f };
	return cv::Point2f(mScreenW * fr[i % 3], mScreenH * fr[i / 3]);
}

void CalibrationSession::ChooseStepDots()
{
	std::vector<int> grid = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	std::shuffle(grid.begin(), grid.end(), mRng);
	mStepDots.assign(grid.begin(), grid.begin() + kDotsPerMovedStep);
}

void CalibrationSession::ShowDots(int active)
{
	std::vector<CalibrationDot> dots;
	for (size_t i = 0; i < mStepDots.size(); i++)
	{
		cv::Point2f position = GridPos(mStepDots[i]);
		dots.push_back({ static_cast<int>(position.x), static_cast<int>(position.y) });
	}
	SetCalibrationDots(dots, active);
	//char buf[200];
	//sprintf_s(buf, "Calibration %d/9: look at the yellow dot\n"
	//				"and move your head slowly around: sideways, closer/farther, a little turn.\n"
	//				"Press Esc to cancel.", mIndex + 1);
	//SetOverlayText(buf);
}

void CalibrationSession::ShowIntro()
{
	ShowDots(-1);

	char message[300];
	sprintf_s(message,
		"KALIBRACIJA\n"
		"Korak 1 od %d: glava stoji skroz mirno.\n"
		"Sedi udobno, otprilike duzinu jedne ruke udaljeno od ekrana.\n"
		"Gledaj u ZELENU tacku, a potom u sledecu ZELENU. Pomeraj samo oci.\n"
		"Pritisni SPACE da pocnes. (Esc prekida kalibraciju)",
		1 + kCalibrationSteps);
	SetOverlayText(message);
}

void CalibrationSession::ShowDotText(bool retry)
{
	ShowDots(mDotInStep);

	char message[300];
	sprintf_s(message, "%sKorak %d od %d - tacka %d od %d\n"
		"Gledaj u ZELENU tacku.\n%s",
		retry ? "Lice nije detektovano - ponovo pokusavam ovu tacku.\n" : "",
		mStep + 1, 1 + kCalibrationSteps,
		mDotInStep + 1, static_cast<int>(mStepDots.size()),
		mStep == 0 ? "Drzi glavu mirno." : "Drzi glavu mirno na ovoj, novoj, poziciji.");
	SetOverlayText(message);
}

void CalibrationSession::ShowWaitMove(const char* warning)
{
	ShowDots(-1);

	const std::string missing = MissingTermsText(mCovered);

	std::string message = warning;
	message += "Korak " + std::to_string(mStep + 1) + " od " + std::to_string(1 + kCalibrationSteps) + "\n";
	message += "Pomeri malo glavu na neku NOVU pozicju.\n";
	if (!missing.empty())
		message += "Jos uvek nije pokriveno: " + missing + ".\nPomeri glavu na neki od navedenih nacina.\n";
	else
		message += "Bilo koji smer je u redu, samo da se razlikuje od prethodnih.\n";
	message += "Onda pritisni SPACE i gledaj u zelene tacke.";

	SetOverlayText(message);
}

void CalibrationSession::BeginDots()
{
	mPhase = Phase::Dots;
	mDotInStep = 0;
	mSampling = false;
	mStepSamples.clear();
	mPhaseStart = std::chrono::steady_clock::now();
	ShowDotText();
}

void CalibrationSession::Start(int screenW, int screenH) 
{
	mScreenW = screenW;
	mScreenH = screenH;
	mStep = 0;
	mDotInStep = 0;
	mSampling = false;
	mCurrent.clear();
	mStepSamples.clear();
	mAll.clear();
	mStepPoses.clear();
	mReport = FitReport();
	mRng.seed(std::random_device{}());
	mStepDots.assign(kNeutralGridDots, kNeutralGridDots + kNeutralDotCount);
	mPhase = Phase::IntroNeutral;
	mActive = true;
	ShowIntro();
	for (int i = 0; i < kPoseTerms; i++)
		mCovered[i] = false;
}

void CalibrationSession::Cancel()
{
	mActive = false;
	ClearCalibrationDots();
}

bool CalibrationSession::Capture(bool hasFeature
								, cv::Point2f feature
								, const HeadPose& pose
								, cv::Point2f target
								, int dotId
								, int settleMs
								, int sampleMs)
{
	using namespace std::chrono;
	auto now = steady_clock::now();
	auto elapsedMs = duration_cast<milliseconds>(now - mPhaseStart).count();

	if (!mSampling)
	{
		if (elapsedMs >= settleMs)
		{
			mSampling = true;
			mCurrent.clear();
			mPhaseStart = now;
		}
		return false;
	}

	if (hasFeature)
		mCurrent.push_back({ feature, pose, target, mStep, dotId });

	if (elapsedMs < sampleMs)
		return false;

	mSampling = false;
	mPhaseStart = now;
	return true;
}

bool CalibrationSession::Update(bool hasFeature
								, cv::Point2f feature
								, const HeadPose& pose
								, bool spacePressed
								, CalibrationConfig& config)
{
	if (!mActive)
		return false;

	if (mPhase == Phase::IntroNeutral)
	{
		if (spacePressed)
			BeginDots();
		return false;
	}

	if (mPhase == Phase::WaitMove)
	{
		if (spacePressed)
		{
			const bool allCovered = AllCovered(mCovered);

			if (!hasFeature)
				ShowWaitMove("Lice nije detektovano - stani ispred kamere.\n");
			else if (!allCovered && !MovesMissingTerm(pose, mStepPoses[0], mCovered))
				ShowWaitMove("Neki drugi pokret.");
			else if (allCovered && !PoseFarEnough(pose, mStepPoses))
				ShowWaitMove("Glava je previse blizu nekoj od prethodnih pozicija.\n");
			else
				BeginDots();
		}
		return false;
	}

	if (!Capture(hasFeature, feature, pose, GridPos(mStepDots[mDotInStep]), mDotInStep, kSettleMs, kSampleMs))
		return false;

	if (mCurrent.size() < kMinSamples)
	{
		ShowDotText(true);
		return false;
	}

	mAll.insert(mAll.end(), mCurrent.begin(), mCurrent.end());
	mStepSamples.insert(mStepSamples.end(), mCurrent.begin(), mCurrent.end());
	mDotInStep++;
	
	if (mDotInStep < static_cast<int>(mStepDots.size()))
	{
		ShowDotText();
		return false;
	}

	mStepPoses.push_back(MeanPose(mStepSamples));
	if (mStep > 0)
		UpdateCoverage(mStepPoses[0], mStepPoses.back(), mCovered);
	mStep++;

	if (mStep <= kCalibrationSteps)
	{
		ChooseStepDots();
		mPhase = Phase::WaitMove;
		ShowWaitMove("");
		return false;
	}

	mActive = false;
	ClearCalibrationDots();

	if (!FitCalibration(mAll, kCalibrationSteps + 1, config, &mReport))
	{
		SetOverlayText("Kalibracija neuspesna - pritisni F9 da pokusas ponovo", 5000);
		return false;
	}

	std::ofstream log("calibration_log.txt", std::ios::app);
	log << "neutral step: used " << mReport.used << " of " << mReport.total
		<< " samples, mean error " << mReport.meanErr << " px\n";
	for (int k = 0; k < kNeutralDotCount; ++k)
		log << "  neutral dot " << k << ": mean error " << mReport.neutralDotErr[k] << " px\n";
	log << "head correction: " << (mReport.poseCorrectionOn ? "ON" : "OFF")
		<< ", hold-out error with correction " << mReport.holdOutWith
		<< " px, without " << mReport.holdOutWithout
		<< " px (" << mReport.holdOutCount << " samples)\n";
	log << "movements covered:\n";
	for (int j = 0; j < kPoseTerms; ++j)
		log << "  " << kPoseTermNames[j] << ": " << (mCovered[j] ? "yes" : "NO") << "\n";
	log << "step poses (shiftX, shiftY, scale, yaw, pitch):\n";
	for (size_t i = 0; i < mStepPoses.size(); ++i)
		log << "  step " << i << ": " << mStepPoses[i].shiftX << ", " << mStepPoses[i].shiftY << ", "
		<< mStepPoses[i].scale << ", " << mStepPoses[i].yaw << ", " << mStepPoses[i].pitch << "\n";

	return true;
}