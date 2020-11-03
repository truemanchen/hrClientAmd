
#include "RPPG.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include<dlib/opencv.h>



using namespace cv::dnn;
using namespace std;


#define LOW_BPM 42
#define HIGH_BPM 180
#define REL_MIN_FACE_SIZE 0.4
#define SEC_PER_MIN 60
#define MAX_CORNERS 5
#define MIN_CORNERS 3
#define QUALITY_LEVEL 0.01
#define MIN_DISTANCE 25
#define MAXNUMBERS 3
dlib::shape_predictor sp;
dlib::full_object_detection shape, shape1;
string modelpath = "shape_predictor_68_face_landmarks.dat";



bool RPPG::load(
	const int width, const int height, const double timeBase, const int downsample,
	const double samplingFrequency, const double rescanFrequency,
	const int minSignalSize, const int maxSignalSize,
	const int chunkSize,
	const string dnnProtoPath, const string dnnModelPath,
	const bool gui) {

	this->rPPGAlg = g;
	this->faceDetAlg = deep;
	this->guiMode = gui;
	this->lastSamplingTime = 0;
	this->minFaceSize = cv::Size(min(width, height) * REL_MIN_FACE_SIZE, min(width, height) * REL_MIN_FACE_SIZE);
	this->maxSignalSize = maxSignalSize;
	this->minSignalSize = minSignalSize;
	this->rescanFlag = false;
	this->rescanFrequency = rescanFrequency;
	this->samplingFrequency = samplingFrequency;
	this->timeBase = timeBase;
	this->chunkSize=chunkSize;
	// Load classifier

	dnnClassifier = readNetFromCaffe(dnnProtoPath, dnnModelPath);

	//if (logMode) {
	//	// Setting up logfilepath
	//	std::ostringstream path_1;
	//	path_1 << logPath << "_rppg=" << rPPGAlg << "_facedet=" << faceDetAlg << "_min=" << minSignalSize << "_max=" << maxSignalSize << "_ds=" << downsample;
	//	this->logfilepath = path_1.str();

	//	// Logging bpm according to sampling frequency
	//	std::ostringstream path_2;
	//	path_2 << logfilepath << "_bpm.csv";
	//	logfile.open(path_2.str());
	//	logfile << "time;face_valid;mean;min;max\n";
	//	logfile.flush();

	//	// Logging bpm detailed
	//	std::ostringstream path_3;
	//	path_3 << logfilepath << "_bpmAll.csv";
	//	logfileDetailed.open(path_3.str());
	//	logfileDetailed << "time;face_valid;bpm\n";
	//	logfileDetailed.flush();
	//}


	//initial skindetect
	//SkinInit(skinDetector, "");
	dlib::deserialize(modelpath) >> sp;
	frameCount = 0;

	return true;
}

void RPPG::exit() {
	logfile.close();
	logfileDetailed.close();
}

cv::Mat RPPG::HSV_detector(cv::Mat src)
{
	cv::Mat hsv_image;
	int h = 0;
	int s = 1;
	int v = 2;
	cv::cvtColor(src, hsv_image, 40); //首先转换成到HSV空间
	cv::Mat output_mask = cv::Mat::zeros(src.size(), CV_8UC1);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			uchar* p_mask = output_mask.ptr<uchar>(i, j);
			uchar* p_src = hsv_image.ptr<uchar>(i, j);
			//if (p_src[h] >= 7 && p_src[h] <= 20 && p_src[s] >= 48 && p_src[v] >= 50)
			if (p_src[h] >= 3 && p_src[h] <= 25 && p_src[s] >= 10 && p_src[s] <= 220 && p_src[v] >= 100)

			{
				p_mask[0] = 255;
			}
		}
	}
	cv::Mat detect;
	src.copyTo(detect, output_mask);;
	return output_mask;
}
/* COMMON FUNCTIONS */

double RPPG::getFps(cv::Mat& t, const double timeBase) {

	double result;

	if (t.empty()) {
		result = 1.0;
	}
	else if (t.rows == 1) {
		result = std::numeric_limits<double>::max();
	}
	else {
		double diff = (t.at<int>(t.rows - 1, 0) - t.at<int>(0, 0)) * timeBase;
		result = diff == 0 ? std::numeric_limits<double>::max() : (t.rows - 1) / diff;
	}

	return result;
}

void RPPG::push(cv::Mat& m) {
	const int length = m.rows;
	m.rowRange(1, length).copyTo(m.rowRange(0, length - 1));
	m.pop_back();
}



void RPPG::processFrame(cv::Mat frameRGB, cv::Mat frameGray, int time) {
	cv::Mat skinMask[4];
	std::vector<cv::Mat> sMs;
	std::vector<cv::Mat> masks;
	frameCount++;

	this->time = time;

	if (!faceValid) {
		cout << "Not valid, finding a new face" << endl;

		lastScanTime = time;
		detectFace(frameRGB, frameGray);


	}
	else if ((time - lastScanTime) * timeBase >= 1 / rescanFrequency) {

		cout << "Valid, but rescanning face" << endl;

		lastScanTime = time;

		detectFace(frameRGB, frameGray);
		rescanFlag = true;

	}
	else {

		cout << "Tracking face" << endl;

		trackFace(frameGray);


	}

	if (faceValid && (!box.empty())) {
		std::vector<cv::Point> landMark;
		box = box & cv::Rect(0, 0, frameRGB.cols, frameRGB.rows);
		frameRGB(box).copyTo(smallframe);
		dlib::cv_image<dlib::rgb_pixel> dlib_frame(smallframe);
		dlib::rectangle dlibRect(0, 0,
			0 + smallframe.cols, 0 + smallframe.rows);
		shape = sp(dlib_frame, dlibRect);
		for (int i = 0; i < 68; i++) {
			cv::Point temp = cv::Point(shape.part(i).x(), shape.part(i).y());
			landMark.push_back(temp);
		}
		cv::Rect smallbox = cv::Rect(cv::Point(max(0, box.tl().x), max(0, box.tl().y)),
			cv::Point(box.tl().x + box.width, box.tl().y + box.height));
		skinMask[0] = HSV_detector(frameRGB(smallbox));

		cv::imshow("sk0", skinMask[0]);
		makeframedata(skinMask[0], landMark, sMs);
		/*makeframedata(smallframe, landMark, sMs);*/

		// Update fps
		fps = getFps(t, timeBase);
		// Remove old values from raw signal buffer
		while (s.rows > fps * maxSignalSize) {
			push(s);
			/*push(s1);
			push(s2);
			push(s3);
			push(s4);*/
			push(t);

			push(re);
		}
		assert(s.rows == t.rows && s.rows == re.rows);

		// New values
			/*cv::Scalar means1 = mean(smallframe, sMs[0].empty() ? cv::noArray() : sMs[0]);
			cv::Scalar means2 = mean(smallframe, sMs[1].empty() ? cv::noArray() : sMs[1]);
			cv::Scalar means3 = mean(smallframe, sMs[2].empty() ? cv::noArray() : sMs[2]);
			cv::Scalar means4 = mean(smallframe, sMs[3].empty() ? cv::noArray() : sMs[3]);*/
		cv::Scalar means = mean(smallframe, skinMask[0].empty() ? cv::noArray() : sMs[4]);

		// Add new values to raw signal buffer
		double values[] = { means(0), means(1), means(2) };
		/*double values1[] = { means1(0), means1(1), means1(2) };
		double values2[] = { means2(0), means2(1), means2(2) };
		double values3[] = { means3(0), means3(1), means3(2) };
		double values4[] = { means4(0), means4(1), means4(2) };*/
		s.push_back(cv::Mat(1, 3, CV_64F, values));
		/*s1.push_back(cv::Mat(1, 3, CV_64F, values1));
		s2.push_back(cv::Mat(1, 3, CV_64F, values2));
		s3.push_back(cv::Mat(1, 3, CV_64F, values3));
		s4.push_back(cv::Mat(1, 3, CV_64F, values4));*/
		t.push_back(time);

		// Save rescan flag
		re.push_back(rescanFlag);

		// Update fps
		fps = getFps(t, timeBase);

		// Update band spectrum limits
		low = (int)(s.rows * LOW_BPM / SEC_PER_MIN / fps);
		high = (int)(s.rows * HIGH_BPM / SEC_PER_MIN / fps) + 1;

		// If valid signal is large enough: esticvmate
	//	if (s.rows >= fps * minSignalSize) {

	//		// Filtering

	//		extractSignal_g();

	//		// HR estimation
	//		estimateHeartrate();

	//		// Log
	//		log();
	//	}

	//	if (guiMode) {
	//		draw(frameRGB);
	//	}
	//}
		if (s.rows >= fps * minSignalSize && chunkSizeCount % chunkSize == 0) {

			struct result r;
			r.length = s.rows;
			for (int i = 0; i < s.rows; i++) {
				r.green[i] = s.at<double>(i, 1);
				r.reTrack[i] = re.at<uchar>(i, 0);
			}
			r.fps = fps;
			hrResult.push_back(r);

		}

		rescanFlag = false;

		frameGray.copyTo(lastFrameGray);
		/*boxroi.pop_back();*/
	}
}

	/*feather_amount.width = feather_amount.height = (int)cv::norm(points_ann[0] - points_ann[6]) / 8;*/

	void RPPG::detectFace(cv::Mat frameRGB, cv::Mat frameGray) {

		std::cout << "scanning for faces" << endl;
		std::vector<cv::Rect> boxes = {};
		switch (faceDetAlg) {
		case haar:
			// Detect faces with Haar classifier
			haarClassifier.detectMultiScale(frameGray, boxes, 1.1, 2, cv::CASCADE_SCALE_IMAGE, minFaceSize);
			break;
		case deep:
			// Detect faces with DNN
			cv::Mat resize300;
			cv::resize(frameRGB, resize300, cv::Size(300, 300));
			cv::Mat blob = blobFromImage(resize300, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));
			dnnClassifier.setInput(blob);
			cv::Mat detection = dnnClassifier.forward();
			cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
			float confidenceThreshold = 0.6;
			for (int i = 0; i < detectionMat.rows; i++) {
				float confidence = detectionMat.at<float>(i, 2);
				if (confidence > confidenceThreshold) {
					int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frameRGB.cols);
					int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frameRGB.rows);
					int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frameRGB.cols);
					int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frameRGB.rows);
					cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
						(int)(xRightTop - xLeftBottom),
						(int)(yRightTop - yLeftBottom));
					boxes.push_back(object);
				}
			}
			break;
		}

		if (boxes.size() > 0) {

			cout << "Found a face" << endl;


			setNearestBox(boxes);
			detectCorners(frameGray);
			//updateROI();
			//updateMask(frameGray);
			faceValid = true;


		}
		else {

			cout << "Found no face" << endl;
			invalidateFace();
		}
	}



	void RPPG::setNearestBox(std::vector<cv::Rect> boxes) {
		int index = 0;
		cv::Point p = box.tl() - boxes.at(0).tl();//the top-left corner
		int min = p.x * p.x + p.y * p.y;
		for (int i = 1; i < boxes.size(); i++) {
			p = box.tl() - boxes.at(i).tl();
			int d = p.x * p.x + p.y * p.y;
			if (d < min) {
				min = d;
				index = i;
			}
		}
		box = boxes.at(index);
	}

	void RPPG::detectCorners(cv::Mat frameGray) {

		// Define tracking region
		cv::Mat trackingRegion = cv::Mat::zeros(frameGray.rows, frameGray.cols, CV_8UC1);
		cv::Point points[1][4];

		points[0][0] = cv::Point(box.tl().x,
			box.tl().y);
		points[0][1] = cv::Point(box.tl().x + box.width,
			box.tl().y);
		points[0][2] = cv::Point(box.tl().x + box.width,
			box.tl().y + box.height);
		points[0][3] = cv::Point(box.tl().x,
			box.tl().y + box.height);
		const cv::Point* pts[1] = { points[0] };
		int npts[] = { 4 };
		fillPoly(trackingRegion, pts, npts, 1, cv::Scalar(255, 255, 255));//as corners mask;

		// Apply corner detection
		goodFeaturesToTrack(frameGray,
			corners,
			MAX_CORNERS,
			QUALITY_LEVEL,
			MIN_DISTANCE,
			trackingRegion,
			3,
			false,
			0.04);
	}

	void RPPG::trackFace(cv::Mat frameGray) {

		// Make sure enough corners are available
		if (corners.size() < MIN_CORNERS) {
			detectCorners(frameGray);
		}

		Contour2f corners_1;
		Contour2f corners_0;
		std::vector<uchar> cornersFound_1;
		std::vector<uchar> cornersFound_0;
		cv::Mat err;

		// Track face features with Kanade-Lucas-Tomasi (KLT) algorithm
		calcOpticalFlowPyrLK(lastFrameGray, frameGray, corners, corners_1, cornersFound_1, err);

		// Backtrack once to make it more robust
		calcOpticalFlowPyrLK(frameGray, lastFrameGray, corners_1, corners_0, cornersFound_0, err);

		// Exclude no-good corners
		Contour2f corners_1v;
		Contour2f corners_0v;
		for (size_t j = 0; j < corners.size(); j++) {
			if (cornersFound_1[j] && cornersFound_0[j]
				&& norm(corners[j] - corners_0[j]) < 2) {
				corners_0v.push_back(corners_0[j]);
				corners_1v.push_back(corners_1[j]);
			}
			else {
				cout << "Mis!" << std::endl;
			}
		}

		if (corners_1v.size() >= MIN_CORNERS) {

			// Save updated features
			corners = corners_1v;

			// Esticv::Mate affine transform
			cv::Mat transform = estimateRigidTransform(corners_0v, corners_1v, false);//partial affine transform
			/*cv::Mat transform = cv::getAffineTransform(corners_1v, corners_0v);*/
			if (transform.total() > 0) {

				// Update box
				Contour2f boxCoords;
				boxCoords.push_back(box.tl());
				boxCoords.push_back(box.br());
				Contour2f transformedBoxCoords;
				cv::transform(boxCoords, transformedBoxCoords, transform);
				box = cv::Rect(transformedBoxCoords[0], transformedBoxCoords[1]);
				box.tl() = cv::Point(max(0, box.tl().x), max(0, box.tl().y));

			}

		}
		else {
			cout << "Tracking failed! Not enough corners left." << endl;
			invalidateFace();
		}
	}


	void RPPG::invalidateFace() {

		s = cv::Mat1d();
		/*s1 = cv::Mat1d();
		s2 = cv::Mat1d();
		s3 = cv::Mat1d();*/
		/*s_f1 = cv::Mat1d();
		s_f2 = cv::Mat1d();
		s_f3 = cv::Mat1d();
		s_f4 = cv::Mat1d();*/
		t = cv::Mat1d();
		re = cv::Mat1b();
		/*powerSpectrum1 = cv::Mat1d();
		powerSpectrum2 = cv::Mat1d();
		powerSpectrum3 = cv::Mat1d();
		powerSpectrum4 = cv::Mat1d();*/
		faceValid = false;
	}

	//void RPPG::extractSignal_pca() {
	//
	//    // Denoise signals
	//    cv::Mat s_den = cv::Mat(s.rows, s.cols, CV_64F);
	//    denoise(s, re, s_den);
	//
	//    // Normalize signals
	//    normalization(s_den, s_den);
	//
	//    // Detrend
	//    cv::Mat s_det = cv::Mat(s.rows, s.cols, CV_64F);
	//    detrend(s_den, s_det, fps);
	//
	//    // PCA to reduce dimensionality
	//    cv::Mat s_pca = cv::Mat(s.rows, 1, CV_32F);
	//    cv::Mat pc = cv::Mat(s.rows, s.cols, CV_32F);
	//    pcaComponent(s_det, s_pca, pc, low, high);
	//
	//    // Moving average
	//    cv::Mat s_mav = cv::Mat(s.rows, 1, CV_32F);
	//    movingAverage(s_pca, s_mav, 3, fmax(floor(fps/6), 2));
	//	/*bandpass(s_mav, s_mav, low, high);*/
	//
	//    s_mav.copyTo(s_f);
	//
	//    // Logging
	//    if (logMode) {
	//        std::ofstream log;
	//        std::ostringstream filepath;
	//        filepath << logfilepath << "_signal_" << time << ".csv";
	//        log.open(filepath.str());
	//        log << "re;r;g;b;r_den;g_den;b_den;r_det;g_det;b_det;pc1;pc2;pc3;s_pca;s_mav\n";
	//        for (int i = 0; i < s.rows; i++) {
	//            log << re.at<bool>(i, 0) << ";";
	//            log << s.at<double>(i, 0) << ";";
	//            log << s.at<double>(i, 1) << ";";
	//            log << s.at<double>(i, 2) << ";";
	//            log << s_den.at<double>(i, 0) << ";";
	//            log << s_den.at<double>(i, 1) << ";";
	//            log << s_den.at<double>(i, 2) << ";";
	//            log << s_det.at<double>(i, 0) << ";";
	//            log << s_det.at<double>(i, 1) << ";";
	//            log << s_det.at<double>(i, 2) << ";";
	//            log << pc.at<double>(i, 0) << ";";
	//            log << pc.at<double>(i, 1) << ";";
	//            log << pc.at<double>(i, 2) << ";";
	//            log << s_pca.at<double>(i, 0) << ";";
	//            log << s_mav.at<double>(i, 0) << "\n";
	//        }
	//        log.close();
	//    }
	//}
	//
	//
	//void RPPG::extractSignal_xminay() {
	//
	//    // Denoise signals
	//    cv::Mat s_den = cv::Mat(s.rows, s.cols, CV_64F);
	//    denoise(s, re, s_den);
	//
	//    // Normalize raw signals
	//    cv::Mat s_n = cv::Mat(s_den.rows, s_den.cols, CV_64F);
	//    normalization(s_den, s_n);
	//
	//    // Calculate X_s signal
	//    cv::Mat x_s = cv::Mat(s.rows, s.cols, CV_64F);
	//    addWeighted(s_n.col(0), 3, s_n.col(1), -2, 0, x_s);
	//
	//    // Calculate Y_s signal
	//    cv::Mat y_s = cv::Mat(s.rows, s.cols, CV_64F);
	//    addWeighted(s_n.col(0), 1.5, s_n.col(1), 1, 0, y_s);
	//    addWeighted(y_s, 1, s_n.col(2), -1.5, 0, y_s);
	//
	//    // Bandpass
	//    cv::Mat x_f = cv::Mat(s.rows, s.cols, CV_32F);
	//    bandpass(x_s, x_f, low, high);
	//    x_f.convertTo(x_f, CV_64F);
	//    cv::Mat y_f = cv::Mat(s.rows, s.cols, CV_32F);
	//    bandpass(y_s, y_f, low, high);
	//    y_f.convertTo(y_f, CV_64F);
	//
	//    // Calculate alpha
	//    cv::Scalar mean_x_f;
	//    cv::Scalar stddev_x_f;
	//    meanStdDev(x_f, mean_x_f, stddev_x_f);
	//    cv::Scalar mean_y_f;
	//    cv::Scalar stddev_y_f;
	//    meanStdDev(y_f, mean_y_f, stddev_y_f);
	//    double alpha = stddev_x_f.val[0]/stddev_y_f.val[0];
	//
	//    // Calculate signal
	//    cv::Mat xminay = cv::Mat(s.rows, 1, CV_64F);
	//    addWeighted(x_f, 1, y_f, -alpha, 0, xminay);
	//
	//    // Moving average
	//    movingAverage(xminay, s_f, 3, fmax(floor(fps/6), 2));
	//
	//    // Logging
	//    if (logMode) {
	//        std::ofstream log;
	//        std::ostringstream filepath;
	//        filepath << logfilepath << "_signal_" << time << ".csv";
	//        log.open(filepath.str());
	//        log << "r;g;b;r_den;g_den;b_den;x_s;y_s;x_f;y_f;s;s_f\n";
	//        for (int i = 0; i < s.rows; i++) {
	//            log << s.at<double>(i, 0) << ";";
	//            log << s.at<double>(i, 1) << ";";
	//            log << s.at<double>(i, 2) << ";";
	//            log << s_den.at<double>(i, 0) << ";";
	//            log << s_den.at<double>(i, 1) << ";";
	//            log << s_den.at<double>(i, 2) << ";";
	//            log << x_s.at<double>(i, 0) << ";";
	//            log << y_s.at<double>(i, 0) << ";";
	//            log << x_f.at<double>(i, 0) << ";";
	//            log << y_f.at<double>(i, 0) << ";";
	//            log << xminay.at<double>(i, 0) << ";";
	//            log << s_f.at<double>(i, 0) << "\n";
	//        }
	//        log.close();
	//    }
	//}
	//void RPPG::findMaxInd(int start,int end) {
	//	maxind.clear();
	//	for (int i = start; i < end; i++) {
	//		if (maxnum==vote[i]) {
	//			maxind.push_back(i);
	//		}
	//	}
	//}
	void RPPG::makeframedata(cv::Mat frameRGB, std::vector<cv::Point>ptr, std::vector<cv::Mat> & processframe)
	{
		cv::Mat src = frameRGB.clone();
		std::vector<cv::Point>ptr1;
		std::vector<cv::Point>ptr2;
		std::vector<cv::Point>ptr3;
		std::vector<cv::Point>ptr4;
		std::vector<cv::Point>ptr5;
		std::vector<cv::Mat> masks;
		cv::Mat mask1 = cv::Mat::zeros(frameRGB.size(), CV_8UC1);
		cv::Mat mask2 = mask1.clone();
		cv::Mat mask3 = mask1.clone();
		cv::Mat mask4 = mask1.clone();
		cv::Mat mask5 = mask1.clone();

		//left cheek
		cv::Point2i distance = ptr[29] - ptr[27];
		for (int i = 0; i < 9; i++)ptr1.push_back(ptr[i]);
		ptr1.push_back(ptr[48]);
		ptr1.push_back(ptr[31]);
		ptr1.push_back(ptr[27]);
		ptr1.push_back(ptr[21]);
		ptr1.push_back(ptr[20]);
		ptr1.push_back(ptr[19]);
		ptr1.push_back(ptr[18]);


		//nose
		ptr2.push_back(ptr[27]);
		ptr2.push_back(ptr[31]);
		ptr2.push_back(ptr[35]);

		//right cheek
		for (int i = 8; i < 17; i++)ptr3.push_back(ptr[i]);
		for (int i = 26; i > 21; i--)ptr3.push_back(ptr[i]);
		ptr3.push_back(ptr[27]);
		ptr3.push_back(ptr[35]);
		ptr3.push_back(ptr[54]);

		//forehead
		for (int i = 17; i < 27; i++)ptr4.push_back(ptr[i]);
		cv::Point2i noselength = 1 * (ptr[27] - ptr[30]);
		ptr4.push_back(ptr[26] + noselength);
		ptr4.push_back(ptr[17] + noselength);

		//lip
		for (int i = 48; i < 60; i++)ptr5.push_back(ptr[i]);


		std::vector<std::vector<cv::Point> > vpts1;
		std::vector<std::vector<cv::Point> > vpts2;
		std::vector<std::vector<cv::Point> > vpts3;
		std::vector<std::vector<cv::Point> > vpts4;
		std::vector<std::vector<cv::Point> > vpts5;

		vpts1.push_back(ptr1);
		//vpts1.push_back(ptr3);
		//vpts1.push_back(ptr2);
		vpts1.push_back(ptr4);
		//left cheek+right cheek
		vpts2.push_back(ptr2);//nose

		vpts3.push_back(ptr3);
		//lip
		vpts4.push_back(ptr4);//forehead
		vpts5.push_back(ptr1);
		vpts5.push_back(ptr3);
		vpts5.push_back(ptr4);


		fillPoly(mask1, vpts1, cv::Scalar(255, 255, 255), 8, 0);
		masks.push_back(mask1);
		fillPoly(mask2, vpts2, cv::Scalar(255, 255, 255), 8, 0);
		masks.push_back(mask2);
		fillPoly(mask3, vpts3, cv::Scalar(255, 255, 255), 8, 0);
		masks.push_back(mask3);
		fillPoly(mask4, vpts4, cv::Scalar(255, 255, 255), 8, 0);
		masks.push_back(mask4);

		fillPoly(mask5, vpts5, cv::Scalar(255, 255, 255), 8, 0);
		masks.push_back(mask5);

		cv::Mat final1 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);
		cv::Mat final2 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);
		cv::Mat final3 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);
		cv::Mat final4 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);
		cv::Mat final5 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);

		bitwise_and(src, src, final1, mask1);
		bitwise_and(src, src, final2, mask2);
		bitwise_and(src, src, final3, mask3);
		bitwise_and(src, src, final4, mask4);
		bitwise_and(src, src, final5, mask5);


		processframe.push_back(final1);
		processframe.push_back(final2);
		processframe.push_back(final3);
		processframe.push_back(final4);
		processframe.push_back(final5);

		cv::imshow("area1", final1);
		cv::imshow("area2", final2);
		cv::imshow("area3", final3);
		cv::imshow("area4", final4);
		cv::imshow("area5", final5);

	}
	double RPPG::findDistance(cv::Mat powerSpectrum) {
		double x, y, distance;
		int dis1, dis2;
		dis1 = inds[0];
		dis2 = inds[1];
		if (pow((dis1 - dis2), 2) != 1) {
			y = powerSpectrum.at<double>(dis2, 0);
			x = powerSpectrum.at<double>(dis1, 0);
		}
		else {
			y = powerSpectrum.at<double>(inds[2], 0);
			x = powerSpectrum.at<double>(dis1, 0);
		}
		cout << "x= " << x << endl;
		cout << "y= " << y << endl;
		distance = pow((x - y), 2);
		return distance;
	}
