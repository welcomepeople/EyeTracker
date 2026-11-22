#pragma once
#include <string>

constexpr int kPoseTerms = 5;

struct CalibrationConfig {
	int screenW = 0;
	int screenH = 0;
	bool valid = false;
	double coeffsX[6] = {};	//koef za mapiranje pogleda
	double coeffsY[6] = {}; //koef za mapiranje pogleda
	bool calibrated = false;
	bool hasPose = false;
	double poseRef[kPoseTerms] = {};
	double poseX[kPoseTerms] = {};
	double poseY[kPoseTerms] = {};
};

bool LoadConfig(const std::string& path, CalibrationConfig& config);
bool SaveConfig(const std::string& path, const CalibrationConfig& config);
std::string GetConfigPath();