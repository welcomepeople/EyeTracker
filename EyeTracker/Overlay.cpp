#include "Overlay.h"
#include "Constants.h"

static std::atomic<int> gGazeX{ 0 };
static std::atomic<int> gGazeY{ 0 };
static HBRUSH bgBrush = CreateSolidBrush(TRANSPARENT_COLOR);
static HBRUSH redBrush = CreateSolidBrush(RGB(255, 0, 0));

void SetGazePoint(int x, int y) {
	gGazeX.store(x);
	gGazeY.store(y);
}

void PaintGazeMarker(HDC hdc, RECT clientRect) {

    //int centerX = (rect.right - rect.left) / 2;
    //int centerY = (rect.bottom - rect.top) / 2;
    //int radius = 10;

	int x = gGazeX.load();
	int y = gGazeY.load();
	int radius = 10;

    // crveni kruzic
    HBRUSH oldBrush = (HBRUSH)SelectObject(hdc, redBrush);

    //HPEN redPen = (HPEN)CreatePen(PS_SOLID, 3, RGB(255, 0, 0));
    //HPEN oldPen = (HPEN)SelectObject(hdc, redPen);

    Ellipse(hdc, x - radius, y - radius, x + radius, y + radius);

    SelectObject(hdc, oldBrush);
    //SelectObject(hdc, oldPen);
    //DeleteObject(redPen);
}

void PaintBackground(HDC hdc, RECT clientRect) {
    FillRect(hdc, &clientRect, bgBrush);
}

void CleanupOverlay() {
    DeleteObject(redBrush);
    DeleteObject(bgBrush);
}