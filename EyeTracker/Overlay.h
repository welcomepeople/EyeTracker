#pragma once
#include "framework.h"
#include <string>

void SetGazePoint(int x, int y);
void PaintGazeMarker(HDC hdc, RECT clientRect);
void PaintBackground(HDC hdc, RECT clientRect);
void CleanupOverlay();

void SetCalibrationDot(int x, int y, bool visible);
void SetOverlayText(const std::string& text, int durationMs = 0);
void PaintCalibration(HDC hdc, RECT clientRect);