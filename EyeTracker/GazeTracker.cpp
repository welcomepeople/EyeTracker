#include "GazeTracker.h"
#include "Overlay.h"
#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/array2d.h>
#include <dlib/pixel.h>

#include <chrono>
#include "Config.h"
#include "Calibration.h"

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

static cv::Point2f FindPupil(const cv::Mat& grayFrame
							, cv::Mat& debugFrame
							, const dlib::full_object_detection& shape
							, int startIdx
							, int endIdx
							, double scale = 1.0
							, const std::string& debugLabel = "") {

	std::vector<cv::Point> poly;
	for (int i = startIdx; i <= endIdx; i++)
		poly.emplace_back(cvRound(shape.part(i).x() / scale)
						, cvRound(shape.part(i).y() / scale));
		
	int minX = poly[0].x;
	int maxX = minX;
	int minY = poly[0].y;
	int maxY = minY;

	for (const auto& p : poly) {
		minX = (std::min)(minX, p.x);
		maxX = (std::max)(maxX, p.x);
		minY = (std::min)(minY, p.y);
		maxY = (std::max)(maxY, p.y);
	}

	int eyeWidth = maxX - minX;

	if (eyeWidth < 8)		//zasto bas 8?
		return cv::Point2f(-1, -1);

	cv::Rect roi(minX - 3, minY - 3, eyeWidth + 6, (maxY - minY) + 6);
	roi &= cv::Rect(0, 0, grayFrame.cols, grayFrame.rows);
	if (roi.width < 8 || roi.height < 4)
		return cv::Point2f(-1, -1);

	//poligon iscrtan unutar oka
	std::vector<std::vector<cv::Point>> drawPoly{ poly };
	cv::polylines(debugFrame, drawPoly, true, cv::Scalar(255, 255, 0), 1);
	
	#pragma region Maska
	//maska, izdvoj unutrasnjost oka
	cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8U);
	std::vector<cv::Point>local;
	for (const auto& p : poly)
		local.push_back(p-roi.tl());
	std::vector<std::vector<cv::Point>> localPoly{ local };
	cv::fillPoly(mask, localPoly, cv::Scalar(255));

	int er = (std::max)(1, cvRound(eyeWidth * 0.06));
	cv::erode(
		mask
		, mask
		, cv::getStructuringElement(cv::MORPH_ELLIPSE
									, cv::Size(2 * er + 1, 2 * er + 1))
	);

	int cutL = std::clamp(cvRound(minX + 0.15 * eyeWidth) - roi.x, 0, mask.cols);
	int cutR = std::clamp(cvRound(maxX - 0.15 * eyeWidth) - roi.x, 0, mask.cols);
	if (cutL > 0)
		mask.colRange(0, cutL).setTo(0);
	if (cutR < mask.cols)
		mask.colRange(cutR, mask.cols).setTo(0);

	int maskPixels = cv::countNonZero(mask);
	if (maskPixels < 30)
		return cv::Point2f(-1, -1);
	#pragma endregion Maska

	#pragma region Threshold
	cv::Mat blurred;
	cv::GaussianBlur(grayFrame(roi), blurred, cv::Size(5, 5), 0);

	std::vector<uchar> vals;
	vals.reserve(maskPixels);
	for (int i = 0; i < blurred.rows; i++)
	{
		const uchar* b = blurred.ptr<uchar>(i);
		const uchar* m = mask.ptr<uchar>(i);
		for (int j = 0; j < blurred.cols; j++)
			if (m[j]) vals.push_back(b[j]);
	}
	std::nth_element(vals.begin(), vals.begin() + vals.size() / 5, vals.end());
	uchar thr = vals[vals.size() / 5];

	cv::Mat dark;
	cv::compare(blurred, thr, dark, cv::CMP_LE);
	cv::bitwise_and(dark, mask, dark);
	cv::morphologyEx(
					dark
					, dark
					, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3))
	);
	#pragma endregion Threshold

	#pragma region Selection
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(dark, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	double minArea = 0.004 * eyeWidth * eyeWidth;
	int best = -1;
	double bestScore = -1.0;

	for (int i = 0; i < static_cast<int>(contours.size()); i++)
	{
		double area = cv::contourArea(contours[i]);
		double per = cv::arcLength(contours[i], true);
		if (area < minArea || per == 0)
			continue;
		double score = area * (4 * CV_PI * area / (per * per));
		if (score > bestScore)
		{
			bestScore = score;
			best = i;
		}
	}
	if (best < 0)
		return cv::Point2f(-1, -1);
	#pragma endregion Selection

	#pragma region
	cv::Mat blob = cv::Mat::zeros(dark.size(), CV_8U);
	cv::drawContours(blob, contours, best, cv::Scalar(255), cv::FILLED);

	double sw = 0, sx = 0, sy = 0;
	for (int i = 0; i < blob.rows; i++)
	{
		const uchar* bl = blob.ptr<uchar>(i);
		const uchar* g = blurred.ptr<uchar>(i);
		for (int j = 0; j < blob.cols; j++)
		{
			if (!bl[j]) continue;
			double w = (std::max)(1.0, double(thr) - g[j] + 1.0);
			sw += w;
			sx += w * j;
			sy += w * i;
		}
	}
	if (sw <= 0)
		return cv::Point2f(-1, -1);

	return cv::Point2f(static_cast<float>(sx / sw) + roi.x
						, static_cast<float>(sy / sw) + roi.y);
	#pragma endregion
}

static cv::Point FindIris(const cv::Mat& grayFrame, cv::Mat& debugFrame, 
	const dlib::full_object_detection& shape, int startIdx, int endIdx, 
	double scale = 1.0, const std::string& debugLabel = "") {

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

	int eyeW = maxX - minX;
	int padX = 3;
	int padY = std::max(4, static_cast<int>(eyeW * 0.2));

	cv::Rect eyeRect(minX - padX, minY - padY, 
		eyeW + padX * 2, (maxY - minY) + padY * 2);
	eyeRect &= cv::Rect(0, 0, grayFrame.cols, grayFrame.rows);

	if (eyeRect.width <= 0 || eyeRect.height <= 0)
		return cv::Point(-1, -1);

	cv::rectangle(debugFrame, eyeRect, cv::Scalar(0, 255, 255), 1);

	cv::Mat blurred, equalized;
	cv::equalizeHist(grayFrame(eyeRect), equalized);
	cv::GaussianBlur(equalized, blurred, cv::Size(5, 5), 0);

	int minR = std::max(3, static_cast<int>(eyeW * 0.15));
	int maxR = std::max(minR + 1, static_cast<int>(eyeW * 0.30));

	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT,
		1,
		blurred.cols,
		60,
		9,
		minR, maxR);

	if (circles.empty())
		return cv::Point(-1, -1);

	const cv::Vec3f& c = circles[0];
	cv::Point center(cvRound(c[0]) + eyeRect.x, cvRound(c[1]) + eyeRect.y);

	cv::circle(debugFrame, center, cvRound(c[2]), cv::Scalar(255, 0, 255), 1);

	return center;
}

static cv::Point2f StabilizePupil(cv::Point2f rawPupil, cv::Point2f& smoothed, int& missStreak, float maxJump, int maxMiss) {
	if (rawPupil.x < 0)
		return smoothed;
	if (smoothed.x < 0) {
		smoothed = rawPupil;
		return smoothed;
	}

	float dx = rawPupil.x - smoothed.x;
	float dy = rawPupil.y - smoothed.y;

	float dist = std::sqrt(dx * dx + dy * dy);

	if (dist <= maxJump) {
		smoothed.x = smoothed.x * 0.5f + rawPupil.x * 0.5f;
		smoothed.y = smoothed.y * 0.5f + rawPupil.y * 0.5f;
		missStreak = 0;
	}
	else {
		missStreak++;
		if (missStreak >= maxMiss) {
			smoothed = rawPupil;
			missStreak = 0;
		}
	}

	return smoothed;
}

static bool ComputeGazeFeature(const dlib::full_object_detection& shape,
	cv::Point2f leftIris, cv::Point2f rightIris, cv::Point2f& out,
	double scale = 1.0) {

	if (leftIris.x < 0 || rightIris.x < 0)
		return false;

	auto pt = [&](int idx) {
		return cv::Point2f(shape.part(idx).x() / scale, shape.part(idx).y() / scale);
	};

	auto eyeFeature = [&](cv::Point2f iris, int cornerA, int cornerB) {
		cv::Point2f a = pt(cornerA);
		cv::Point2f b = pt(cornerB);
		float eyeW = b.x - a.x;
		if (std::abs(eyeW) < 1.0f) eyeW = 1.0f;
		float hx = (iris.x - a.x) / eyeW;
		float vy = (iris.y - (a.y + b.y) * 0.5f) / std::abs(eyeW);
		return cv::Point2f(hx, vy);
	};
	
	cv::Point2f l = eyeFeature(leftIris, 36, 39);
	cv::Point2f r = eyeFeature(rightIris, 42, 45);

	out = cv::Point2f((l.x + r.x) * 0.5f, (l.y + r.y) * 0.5f);
	return true;
}

static bool ComputeHeadPose(const dlib::full_object_detection& shape, double scale, int frameW, int frameH, HeadPose& p)
{
	if (shape.num_parts() != 68) return false;

	cv::Point2f rightEyeOuter(shape.part(36).x() / scale, shape.part(36).y() / scale);
	cv::Point2f leftEyeOuter(shape.part(45).x() / scale, shape.part(45).y() / scale);
	cv::Point2f nose(shape.part(30).x() / scale, shape.part(30).y() / scale);

	cv::Point2f midPoint = (rightEyeOuter + leftEyeOuter) * 0.5f;
	float eyeDistance = (float)cv::norm(leftEyeOuter - rightEyeOuter);

	p.shiftX = (midPoint.x - frameW * 0.5f) / eyeDistance;
	p.shiftY = (midPoint.y - frameH * 0.5f) / eyeDistance;
	p.scale = eyeDistance;
	p.yaw = (nose.x - midPoint.x) / eyeDistance;
	p.pitch = (nose.y - midPoint.y) / eyeDistance;
	p.roll = std::atan2(leftEyeOuter.y - rightEyeOuter.y, leftEyeOuter.x - rightEyeOuter.x);

	return true;
}

static cv::Point2f MedianFeature(const cv::Point2f& sample, std::deque<cv::Point2f>& history, size_t maxN = 9) {
	history.push_back(sample);
	if (history.size() > maxN) history.pop_front();

	std::vector<float> xs, ys;
	for (const auto& p : history) {
		xs.push_back(p.x);
		ys.push_back(p.y);
	}
	std::nth_element(xs.begin(), xs.begin() + xs.size() / 2, xs.end());
	std::nth_element(ys.begin(), ys.begin() + ys.size() / 2, ys.end());
	return cv::Point2f(xs[xs.size() / 2], ys[ys.size() / 2]);
}

static void CaptureLoop(HWND targetWindow) {

	//CalibrationConfig config;
	//std::string configPath = GetConfigPath();
	//if (!LoadConfig(configPath, config))
	//{
	//	config.screenW = GetSystemMetrics(SM_CXSCREEN);
	//	config.screenH = GetSystemMetrics(SM_CYSCREEN);
	//	SaveConfig(configPath, config);
	//}

	RECT rect;
	GetClientRect(targetWindow, &rect);
	int x = (rect.right - rect.left) / 2;
	int y = (rect.bottom - rect.top) / 2;

	cv::VideoCapture cap(0);
	//cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	//cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

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

	cv::Point2f smoothedLPupil(-1, -1);
	cv::Point2f smoothedRPupil(-1, -1);
	int lMissStreak = 0;
	int rMissStreak = 0;
	const float maxJumpDist = 8.0f;
	const int maxMissStreak = 5;

	int framesSinceDetect = 0;
	const int detectingInterval = 10;
	std::future<std::vector<dlib::rectangle>> detectFuture;
	bool detectionInProgress = false;

	std::deque<cv::Point2f> gazeHistory;

	CalibrationConfig config;
	std::string configPath = GetConfigPath();
	LoadConfig(configPath, config);

	int screenW = GetSystemMetrics(SM_CXSCREEN);
	int screenH = GetSystemMetrics(SM_CYSCREEN);
	if (config.screenW != screenW || config.screenH != screenH)
	{
		config.screenW = screenW;
		config.screenH = screenH;
		config.calibrated = false;
	}

	CalibrationSession session;
	bool f9WasPressed = false;
	cv::Point2f dotSmoothed(-1, -1);

	if (!config.calibrated)
		SetOverlayText("Press F9 to calibrate");

	while (gRunning) {
		auto t0 = std::chrono::steady_clock::now();

		bool hasFeature = false;
		cv::Point2f gazeFeature;

		cap >> frame;
		if (frame.empty())
			continue;

		bool f9Press = (GetAsyncKeyState(VK_F9) & 0x8000) != 0;
		if (f9Press && !f9WasPressed && !session.Active())
		{
			session.Start(screenW, screenH);
			SetGazePoint(-1000, -1000); //sakrij crvenu tackicu
		}

		f9WasPressed = f9Press;
		if (session.Active() && (GetAsyncKeyState(VK_ESCAPE) & 0x8000))
		{
			session.Cancel();
			SetOverlayText("Calibration cancelled", 2000);
		}

		double scale = 0.5;
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
			shape = predictor(dlibImage, lastFace);
			
			cv::Mat fullGray;
			cv::cvtColor(frame, fullGray, cv::COLOR_BGR2GRAY);

			cv::Point2f leftPupilRaw = FindPupil(fullGray, frame, shape, 36, 41, scale);
			cv::Point2f rightPupilRaw = FindPupil(fullGray, frame, shape, 42, 47, scale);

			//cv::Point2f leftPupilRaw = FindPupil(fullGray, frame, shape, 36, 41, scale, "Left Eye Threshold");
			//cv::Point2f rightPupilRaw = FindPupil(fullGray, frame, shape, 42, 47, scale, "Right Eye Threshold");
			// 
			//cv::Point leftPupilRaw = FindIris(fullGray, frame, shape, 36, 41, scale, "Left Eye Threshold");
			//cv::Point rightPupilRaw = FindIris(fullGray, frame, shape, 42, 47, scale, "Right Eye Threshold");

			cv::Point2f leftPupil = StabilizePupil(leftPupilRaw, smoothedLPupil, lMissStreak, maxJumpDist, maxMissStreak);
			cv::Point2f rightPupil = StabilizePupil(rightPupilRaw, smoothedRPupil, rMissStreak, maxJumpDist, maxMissStreak);

			hasFeature = ComputeGazeFeature(shape, leftPupil, rightPupil, gazeFeature, scale);

			/*if (hasFeature)
			{
				cv::Point2f smooth = MedianFeature(gazeFeature, gazeHistory);
				char buf[64];
				sprintf_s(buf, "gaze: x=%.2f y=%.2f", gazeFeature.x, gazeFeature.y);
				cv::putText(frame, buf, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
			}*/

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

		if (session.Active())
		{
			if (session.Update(hasFeature, gazeFeature, config))
			{
				char buf[128];
				if (SaveConfig(configPath, config))
					sprintf_s(buf, "Calibration saved (average fit error %.0f px)", session.LastError());
				else
					sprintf_s(buf, "Calibrated, but saving the configuration file failed");
				SetOverlayText(buf, 8000);
				dotSmoothed = cv::Point2f(-1, -1);
			}
		}
		else if (config.calibrated && hasFeature)
		{
			cv::Point2f smooth = MedianFeature(gazeFeature, gazeHistory);
			cv::Point2f target = MapGazeToScreen(config, smooth);

			if (dotSmoothed.x < 0)
				dotSmoothed = target;
			else
				dotSmoothed = dotSmoothed * 0.8f + target * 0.2f;

			SetGazePoint((int)dotSmoothed.x, (int)dotSmoothed.y);
		}

		cv::imshow("Camera Debug", frame);
		cv::waitKey(1);
		
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