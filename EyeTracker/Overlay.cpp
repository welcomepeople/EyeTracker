#include "Overlay.h"
#include "Constants.h"
#include <mutex>

static std::atomic<int> gGazeX{ 0 };
static std::atomic<int> gGazeY{ 0 };
static HBRUSH bgBrush = CreateSolidBrush(TRANSPARENT_COLOR);
static HBRUSH redBrush = CreateSolidBrush(RGB(255, 0, 0));

static std::atomic<bool> calibrationDotVisible{ false };
static std::atomic<int> calibrationDotX{ 0 };
static std::atomic<int> calibrationDotY{ 0 };

static std::mutex calibrationTextMutex;
static std::string calibrationText;
static bool calibrationTextExpires = false;
static std::chrono::steady_clock::time_point calibrationTextExpiry;

void SetGazePoint(int x, int y) 
{
	gGazeX.store(x);
	gGazeY.store(y);
}

void PaintGazeMarker(HDC hdc, RECT clientRect) 
{
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

void PaintBackground(HDC hdc, RECT clientRect) 
{
    FillRect(hdc, &clientRect, bgBrush);
}

void CleanupOverlay() 
{
    DeleteObject(redBrush);
    DeleteObject(bgBrush);
}

void SetCalibrationDot(int x, int y, bool visible)
{
    calibrationDotX = x;
    calibrationDotY = y;
    calibrationDotVisible = visible;
}

void SetOverlayText(const std::string& text, int durationMs)
{
    std::lock_guard<std::mutex> lock(calibrationTextMutex);
    calibrationText = text;
    calibrationTextExpires = durationMs > 0;
    if (calibrationTextExpires)
        calibrationTextExpiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(durationMs);
}

void PaintCalibration(HDC hdc, RECT clientRect)
{
    if (calibrationDotVisible)
    {
        int x = calibrationDotX, y = calibrationDotY, r = 18;
        HBRUSH yellow = CreateSolidBrush(RGB(255, 255, 0));
        HBRUSH oldBrush = (HBRUSH)SelectObject(hdc, yellow);
        HPEN oldPen = (HPEN)SelectObject(hdc, GetStockObject(NULL_PEN));
        Ellipse(hdc, x - r, y - r, x + r, y + r);
        SelectObject(hdc, oldBrush);
        SelectObject(hdc, oldPen);
        DeleteObject(yellow);
    }

    std::string text;
    //da capture thread ne mora da ceka UI thread, lock se otkljuca kad izadje iz scope-a
    {   
        std::lock_guard<std::mutex> lock(calibrationTextMutex);
        if (calibrationTextExpires && std::chrono::steady_clock::now() > calibrationTextExpiry)
            calibrationText.clear();
        text = calibrationText;
    }

    if (!text.empty()) 
    {
        HFONT font = CreateFontA(32, 0, 0, 0, 
                    FW_BOLD, 
                    FALSE, 
                    FALSE, 
                    FALSE, 
                    DEFAULT_CHARSET,
                    OUT_DEFAULT_PRECIS,     
                    CLIP_DEFAULT_PRECIS, 
                    ANTIALIASED_QUALITY, 
                    DEFAULT_PITCH, 
                    "Segoe UI"
        );
        HFONT oldFont = (HFONT)SelectObject(hdc, font);
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, RGB(255, 255, 255));
        RECT r = clientRect;
        r.top = 10;
        DrawTextA(hdc, text.c_str(), -1, &r, DT_CENTER | DT_TOP | DT_WORDBREAK);
        SelectObject(hdc, oldFont);
        DeleteObject(font);
    }
}