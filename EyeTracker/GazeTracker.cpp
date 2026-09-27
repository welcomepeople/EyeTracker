#include "GazeTracker.h"
#include "Overlay.h"
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/array2d.h>
#include <dlib/pixel.h>

#include <chrono>

static std::atomic<bool> gRunning{ false };
static std::thread gCaptureThread;

static dlib::array2d<dlib::bgr_pixel> MatToDlib(const cv::Mat& mat) {

	dlib::array2d<dlib::bgr_pixel> dlibImg(mat.rows, mat.cols);
	for (int r = 0; r < mat.rows; r++) {

		const cv::Vec3b* rowPointer = mat.ptr<cv::Vec3b>(r);
		for (int c = 0; c < mat.cols; c++) {

			cv::Vec3b pixel = rowPointer[c];
			dlibImg[r][c] = dlib::bgr_pixel(pixel[0], pixel[1], pixel[2]);
		}
	}
	return dlibImg;
}

static dlib::array2d<unsigned char> MatToDlibGrayscale(const cv::Mat& mat) {

	cv::Mat gray;
	cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
	
	dlib::array2d<unsigned char> dlibImg(gray.rows, gray.cols);
	for (int r = 0; r < gray.rows; r++) {

		const uchar* rowPointer = gray.ptr<uchar>(r);
		for (int c = 0; c < gray.cols; c++) {

			dlibImg[r][c] = rowPointer[c];
		}
	}
	return dlibImg;
}

static void CaptureLoop(HWND targetWindow) {

	RECT rect;
	GetClientRect(targetWindow, &rect);
	int x = (rect.right - rect.left) / 2;
	int y = (rect.bottom - rect.top) / 2;

	cv::VideoCapture cap(0);
	//cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	//cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

	double actualw = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	double actualh = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

	if (!cap.isOpened()) {
		MessageBoxA(nullptr, "Could not open camera", "Eye Tracker", MB_OK);
		return;
	}

	dlib::frontal_face_detector detector =
		dlib::get_frontal_face_detector();

	dlib::shape_predictor predictor;

	try {
		dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
	}
	catch (std::exception& e) {
		MessageBoxA(nullptr, e.what(), "Failed to load landmark model.", MB_OK);
		return;
	}

	cv::Mat frame, small;
	dlib::full_object_detection shape;
	dlib::array2d<unsigned char> dlibImage;
	std::vector<dlib::rectangle> faces;
	dlib::rectangle lastFace;
	bool hasFace = false;
	int framesSinceDetect = 0;
	const int detectingInterval = 30;

	while (gRunning) {
		auto t0 = std::chrono::steady_clock::now();

		cap >> frame;
		if (frame.empty())
			continue;

		double scale = 0.35;
		cv::resize(frame, small, cv::Size(), scale, scale);

		auto t1 = std::chrono::steady_clock::now();

		dlibImage = MatToDlibGrayscale(small);

		bool needsFullDetect = !hasFace || (framesSinceDetect >= detectingInterval);
		if (needsFullDetect) {
			std::vector<dlib::rectangle> faces2 = detector(dlibImage);
			if (!faces2.empty()) {
				lastFace = faces2[0];
				hasFace = true;
			}
			else {
				hasFace = false;
			}
			framesSinceDetect = 0;
		}
		if (hasFace) {
			dlib::full_object_detection shape = predictor(dlibImage, lastFace);
			
			//long minX = shape.part(0).x();
			//long maxX = minX;
			//long minY = shape.part(0).y();
			//long maxY = minY;

			//for (unsigned int i = 1; i < shape.num_parts(); i++) {
			//	minX = std::min(minX, (long)shape.part(i).x());
			//	maxX = std::max(maxX, (long)shape.part(i).x());
			//	minY = std::min(minY, (long)shape.part(i).y());
			//	maxY = std::max(maxY, (long)shape.part(i).y());
			//}
			//long margin = 20;

			///*long clampedLeft = std::max(0L, minX - margin);
			//long clampedTop = std::max(0L, minY - margin);
			//long clampedRight = std::min(static_cast<long>(dlibImage.nc()) - 1, maxX + margin);
			//long clampedBottom = std::min(static_cast<long>(dlibImage.nr()) - 1, maxY + margin);

			//lastFace = dlib::rectangle(clampedLeft, clampedTop, clampedRight, clampedBottom);*/
			//lastFace = dlib::rectangle(minX - margin, minY - margin, maxX + margin, maxY + margin);

			for (unsigned int i = 36; i <= 47; ++i) {
				cv::Point p(shape.part(i).x() / scale, shape.part(i).y() / scale);
				cv::circle(frame, p, 2, cv::Scalar(0, 255, 0), cv::FILLED);
			}

			framesSinceDetect++;
		}

		cv::imshow("Camera Debug", frame);
		cv::waitKey(1);
		
		SetGazePoint(x, y);
		InvalidateRect(targetWindow, nullptr, FALSE);
	}

	cv::destroyWindow("Camera Debug");
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