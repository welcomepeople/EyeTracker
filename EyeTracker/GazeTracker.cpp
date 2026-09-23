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
	while (gRunning) {
		auto t0 = std::chrono::steady_clock::now();

		cap >> frame;
		if (frame.empty())
			continue;

		double scale = 0.35;
		cv::resize(frame, small, cv::Size(), scale, scale);

		auto t1 = std::chrono::steady_clock::now();

		dlib::array2d<dlib::bgr_pixel> dlibImage = MatToDlib(small);
		auto t2 = std::chrono::steady_clock::now();
		std::vector<dlib::rectangle> faces = detector(dlibImage);
		auto t3 = std::chrono::steady_clock::now();
		if (!faces.empty()) {

			dlib::full_object_detection shape =
				predictor(dlibImage, faces[0]);

			for (unsigned int i = 36; i <= 47; ++i) {
				cv::Point p(shape.part(i).x() / scale, shape.part(i).y() / scale);
				cv::circle(frame, p, 2, cv::Scalar(0, 255, 0), cv::FILLED);
			}
		}
		auto t4 = std::chrono::steady_clock::now();
		auto ms = [](auto a, auto b) {
			return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
			};
		char buf[256];
		sprintf_s(buf, "capture=%lldms convert=%lldms detect=%lldms landmarks=%lldms",
			ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4));
		//sprintf_s(buf, "%.0fx%.0f", actualw, actualh);
		cv::setWindowTitle("Camera Debug", buf);

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