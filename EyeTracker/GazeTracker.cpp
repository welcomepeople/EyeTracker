#include "GazeTracker.h"
#include "Overlay.h"
#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
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

static cv::Point FindPupil(const cv::Mat& grayFrame, cv::Mat& debugFrame, const dlib::full_object_detection& shape, int startIdx, int endIdx, double scale = 1.0, const std::string& debugLabel = "") {

	int minX = shape.part(startIdx).x() / scale;
	int maxX = minX;
	int minY = shape.part(startIdx).y() / scale;
	int maxY = minY;

	for (int i = startIdx + 1; i <= endIdx; i++) {
		int px = shape.part(i).x() / scale;
		int py = shape.part(i).y() / scale;
		minX = std::min(minX, px);
		maxX = std::max(maxX, px);
		minY = std::min(minY, py);
		maxY = std::max(maxY, py);
	}

	//margina kako ivica ne bi odsekla zenice
	int margin = 8;
	cv::Rect eyeRect(minX - margin, minY - margin,
		(maxX - minX) + margin * 2, (maxY - minY) + margin * 2);
	eyeRect &= cv::Rect(0, 0, grayFrame.cols, grayFrame.rows);

	if (eyeRect.width <= 0 || eyeRect.height <= 0)
		return cv::Point(-1, -1);

	cv::rectangle(debugFrame, eyeRect, cv::Scalar(0, 255, 255), 1);

	cv::Mat eyeROI = grayFrame(eyeRect);

	cv::Mat threshold;
	cv::threshold(eyeROI, threshold, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

	/*if (!debugLabel.empty()) {
		cv::Mat enlarged;
		cv::resize(threshold, enlarged, cv::Size(), 10, 10, cv::INTER_NEAREST);
		cv::imshow(debugLabel, enlarged);
	}*/

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(threshold, threshold, cv::MORPH_CLOSE, kernel);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(threshold, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	if (contours.empty())
		return cv::Point(-1, -1);

	auto largest = std::max_element(contours.begin(), contours.end(),
		[](const auto& a, const auto& b) {
			return cv::contourArea(a) < cv::contourArea(b);
		});

	cv::Moments m = cv::moments(*largest);
	if (m.m00 == 0)
		return cv::Point(-1, -1);

	cv::Point pupilInROI(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));

	return cv::Point(pupilInROI.x + eyeRect.x, pupilInROI.y + eyeRect.y);
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
	std::future<std::vector<dlib::rectangle>> detectFuture;
	bool detectionInProgress = false;

	while (gRunning) {
		auto t0 = std::chrono::steady_clock::now();

		cap >> frame;
		if (frame.empty())
			continue;

		double scale = 0.35;
		cv::resize(frame, small, cv::Size(), scale, scale);

		auto t1 = std::chrono::steady_clock::now();

		dlibImage = MatToDlibGrayscale(small);

		//the following comment block is being replaced

		//bool needsFullDetect = !hasFace || (framesSinceDetect >= detectingInterval);
		//if (needsFullDetect) {
		//	std::vector<dlib::rectangle> faces2 = detector(dlibImage);
		//	if (!faces2.empty()) {
		//		lastFace = faces2[0];
		//		hasFace = true;
		//	}
		//	else {
		//		hasFace = false;
		//	}
		//	framesSinceDetect = 0;
		//}

		//end of comment block
		// 
		//block replacing the comment block
		bool shouldStartDetect = !detectionInProgress 
			&& (!hasFace || framesSinceDetect >= detectingInterval);
		if (shouldStartDetect) {
			dlib::array2d<unsigned char> detectImage;
			dlib::assign_image(detectImage, dlibImage);

			detectFuture = std::async(std::launch::async,
				[&detector, img = std::move(detectImage)]() mutable {
					return detector(img);
				});
			detectionInProgress = true;
		}

		if (detectionInProgress && detectFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {

			std::vector<dlib::rectangle> faces2 = detectFuture.get();
			if (!faces2.empty()) {
				lastFace = faces2[0];
				hasFace = true;
			}
			else 
				hasFace = false;
			framesSinceDetect = 0;
			detectionInProgress = false;
		}
		
		if (hasFace) {
			dlib::full_object_detection shape = predictor(dlibImage, lastFace);
			
			cv::Mat fullGray;
			cv::cvtColor(frame, fullGray, cv::COLOR_BGR2GRAY);

			cv::Point leftPupil = FindPupil(fullGray, frame, shape, 36, 41, scale, "Left Eye Threshold");
			cv::Point rightPupil = FindPupil(fullGray, frame, shape, 42, 47, scale, "Right Eye Threshold");

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

			//for (unsigned int i = 36; i <= 47; ++i) {
			//	cv::Point p(shape.part(i).x() / scale, shape.part(i).y() / scale);
			//	cv::circle(frame, p, 2, cv::Scalar(0, 255, 0), cv::FILLED);
			//}
			
			if (leftPupil.x >= 0) {
				//cv::circle(frame, cv::Point(leftPupil.x / scale, leftPupil.y / scale), 3, cv::Scalar(255, 0, 0), cv::FILLED);
				cv::circle(frame, leftPupil, 3, cv::Scalar(255, 0, 0), cv::FILLED);
			}
			if (rightPupil.x >= 0) {
				//cv::circle(frame, cv::Point(rightPupil.x / scale, rightPupil.y / scale), 3, cv::Scalar(255, 0, 0), cv::FILLED);
				cv::circle(frame, rightPupil, 3, cv::Scalar(255, 0, 0), cv::FILLED);
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