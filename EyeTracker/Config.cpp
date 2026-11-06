#include "Config.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <Windows.h>

static void ParseCoeffs(const std::string& text, double* out) 
{
	std::stringstream ss(text);
	std::string item;
	int i = 0;
	while (i < 6 && std::getline(ss, item, ','))
		out[i++] = std::stod(item);
}

static void WriteCoeffs(std::ofstream& out, const char* key, const double* c) 
{
	out << key << "=";
	for (int i = 0; i < 6; i++)
		out << c[i] << (i < 5 ? "," : "\n");
}

bool LoadConfig(const std::string& path, CalibrationConfig& config)
{
	std::ifstream in(path);
	if (!in)
		return false;

	int found = 0;
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
		}
	}
	catch (...)
	{
		config.valid = false;
		return false;
	}

	config.valid = (found == 4);
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