#include "Config.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <Windows.h>

static void ParseCoeffs(const std::string& text, double* out, int count = 6) 
{
	std::stringstream ss(text);
	std::string item;
	int i = 0;
	while (i < count && std::getline(ss, item, ','))
		out[i++] = std::stod(item);
}

static void WriteCoeffs(std::ofstream& out, const char* key, const double* c, int count = 6) 
{
	out << key << "=";
	for (int i = 0; i < count; i++)
		out << c[i] << (i < count - 1 ? "," : "\n");
}

bool LoadConfig(const std::string& path, CalibrationConfig& config)
{
	std::ifstream in(path);
	if (!in)
		return false;

	int found = 0;
	int poseFound = 0;
	std::string line;
	try
	{
		while (std::getline(in, line))
		{
			size_t eq = line.find('=');
			if (eq == std::string::npos) continue;

			std::string key = line.substr(0, eq);
			std::string value = line.substr(eq + 1);

			if (key == "width_px") { config.screenW = std::stoi(value); found++; }
			else if (key == "height_px") { config.screenH = std::stoi(value); found++; }
			else if (key == "coeffs_x") { ParseCoeffs(value, config.coeffsX); found++; }
			else if (key == "coeffs_y") { ParseCoeffs(value, config.coeffsY); found++; }
			else if (key == "calibrated") { config.calibrated = (value == "1"); found++; }
			else if (key == "pose_ref") { ParseCoeffs(value, config.poseRef, kPoseTerms); poseFound++; }
			else if (key == "pose_x") { ParseCoeffs(value, config.poseX, kPoseTerms); poseFound++; }
			else if (key == "pose_y") { ParseCoeffs(value, config.poseY, kPoseTerms); poseFound++; }
		}
	}
	catch (...)
	{
		config.valid = false;
		return false;
	}

	config.hasPose = (poseFound == 3);
	config.valid = (found == 5);
	return config.valid;
}

bool SaveConfig(const std::string& path, const CalibrationConfig& config)
{
	std::ofstream out(path);
	if (!out) return false;

	out << std::setprecision(8);
	out << "width_px=" << config.screenW << "\n";
	out << "height_px=" << config.screenH << "\n";
	WriteCoeffs(out, "coeffs_x", config.coeffsX);
	WriteCoeffs(out, "coeffs_y", config.coeffsY);

	if (config.hasPose)
	{
		WriteCoeffs(out, "pose_ref", config.poseRef, kPoseTerms);
		WriteCoeffs(out, "pose_x", config.poseX, kPoseTerms);
		WriteCoeffs(out, "pose_y", config.poseY, kPoseTerms);
	}

	out << "calibrated=" << (config.calibrated ? 1 : 0) << "\n";
	return true;
}

std::string GetConfigPath() 
{
	char appData[MAX_PATH];
	DWORD n = GetEnvironmentVariableA("APPDATA", appData, MAX_PATH);
	if (n == 0 || n >= MAX_PATH)
		return "config.ini";

	std::string dir = std::string(appData) + "\\EyeTracker";
	CreateDirectoryA(dir.c_str(), nullptr);
	return dir + "\\config.ini";
}