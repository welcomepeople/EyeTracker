#pragma once
#include "framework.h"
#include <string>
#include <vector>

struct CalibrationDot { int x, y; };

void SetGazePoint(int x, int y);
void PaintGazeMarker(HDC hdc, RECT clientRect);
void PaintBackground(HDC hdc, RECT clientRect);
void CleanupOverlay();

void SetCalibrationDots(const std::vector<CalibrationDot>& dots, int activeIndex);
void SetOverlayText(const std::string& text, int durationMs = 0);
void PaintCalibration(HDC hdc, RECT clientRect);
void ClearCalibrationDots();