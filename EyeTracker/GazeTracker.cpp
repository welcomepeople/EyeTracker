#include "GazeTracker.h"
#include "Overlay.h"
#include <opencv2/opencv.hpp>

static std::atomic<bool> gRunning{ false };
static std::thread gCaptureThread;

static void CaptureLoop(HWND targetWindow) {

	RECT rect;
	GetClientRect(targetWindow, &rect);
	int x = (rect.right - rect.left) / 2;
	int y = (rect.bottom - rect.top) / 2;

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		MessageBoxA(nullptr, "Could not open camera", "Eye Tracker", MB_OK);
		return;
	}

	cv::Mat frame;

	while (gRunning) {
		
		cap >> frame;
		if (frame.empty())
			continue;

		cv::imshow("Camera Debug", frame);
		cv::waitKey(1);
		
		SetGazePoint(x, y);
		InvalidateRect(targetWindow, nullptr, FALSE);
	}
}

void StartCapture(HWND targetWindow) {

	gRunning = true;
	gCaptureThread = std::thread(CaptureLoop, targetWindow);
}

void StopCapture() {

	gRunning = false;
	if (gCaptureThread.joinable())
		gCaptureThread.join();
}