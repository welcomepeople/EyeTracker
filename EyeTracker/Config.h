#pragma once
#include <string>

struct CalibrationConfig {
	int screenW = 0;
	int screenH = 0;
	bool valid = false;
	double coeffsX[6] = {};	//koef za mapiranje pogleda
	double coeffsY[6] = {}; //koef za mapiranje pogleda
	bool calibrated = false;
};

bool LoadConfig(const std::string& path, CalibrationConfig& config);
bool SaveConfig(const std::string& path, const CalibrationConfig& config);
std::string GetConfigPath();