#pragma once
#include "framework.h"

void SetGazePoint(int x, int y);
void PaintGazeMarker(HDC hdc, RECT clientRect);
void PaintBackground(HDC hdc, RECT clientRect);
void CleanupOverlay();
