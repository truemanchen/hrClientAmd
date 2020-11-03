

#include "hrClient.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "MeasureData.h"
#include<thread>
#pragma warning(disable: 4996)

#define DEFAULT_RPPG_ALGORITHM "g"
#define DEFAULT_FACEDET_ALGORITHM "deep"
#define DEFAULT_RESCAN_FREQUENCY 1
#define DEFAULT_SAMPLING_FREQUENCY 1
#define DEFAULT_MIN_SIGNAL_SIZE 6
#define DEFAULT_MAX_SIGNAL_SIZE 8
#define DEFAULT_FRAMES          1200
#define DEFAULT_DOWNSAMPLE 1// x means only every xth frame is used
#define WARP 0

#define DNN_PROTO_PATH "deploy.prototxt"
#define DNN_MODEL_PATH "res10_300x300_ssd_iter_140000.caffemodel"
#define FACE_LANDMARK "shape_predictor_68_face_landmarks.dat"//特征点检测的两个文件


void video_make(int n, int& num,  std::deque<Measure<cv::Mat>>& data, const double fps, const int width, const int height,cv::VideoCapture capture);




//string input = cmd_line.get_arg("-i"); // Filepath for offline mode
//
//// algorithm setting
//rPPGAlgorithm rPPGAlg;
//string rppgAlgString = cmd_line.get_arg("-rppg");
//if (rppgAlgString != "") {
//	rPPGAlg = to_rppgAlgorithm(rppgAlgString);
//}
//else {
//	rPPGAlg = to_rppgAlgorithm(DEFAULT_RPPG_ALGORITHM);
//}
//
//cout << "Using rPPG algorithm " << rPPGAlg << "." << endl;
//
//// face detection algorithm setting
//faceDetAlgorithm faceDetAlg;
//string faceDetAlgString = cmd_line.get_arg("-facedet");
//if (faceDetAlgString != "") {
//	faceDetAlg = to_faceDetAlgorithm(faceDetAlgString);
//}
//else {
//	faceDetAlg = to_faceDetAlgorithm(DEFAULT_FACEDET_ALGORITHM);
//}
//
//cout << "Using face detection algorithm " << faceDetAlg << "." << endl;
//
//// rescanFrequency setting
//double rescanFrequency;
//string rescanFrequencyString = cmd_line.get_arg("-r");
//if (rescanFrequencyString != "") {
//	rescanFrequency = atof(rescanFrequencyString.c_str());
//}
//else {
//	rescanFrequency = DEFAULT_RESCAN_FREQUENCY;
//}
//
//// samplingFrequency setting
//double samplingFrequency;
//string samplingFrequencyString = cmd_line.get_arg("-f").c_str();
//if (samplingFrequencyString != "") {
//	samplingFrequency = atof(samplingFrequencyString.c_str());
//}
//else {
//	samplingFrequency = DEFAULT_SAMPLING_FREQUENCY;
//}
//
//// max signal size setting
//int maxSignalSize;
//string maxSignalSizeString = cmd_line.get_arg("-max");
//if (maxSignalSizeString != "") {
//	maxSignalSize = atof(maxSignalSizeString.c_str());
//}
//else {
//	maxSignalSize = DEFAULT_MAX_SIGNAL_SIZE;
//}
//
//// min signal size setting
//int minSignalSize;
//string minSignalSizeString = cmd_line.get_arg("-min");
//if (minSignalSizeString != "") {
//	minSignalSize = atof(minSignalSizeString.c_str());
//}
//else {
//	minSignalSize = DEFAULT_MIN_SIGNAL_SIZE;
//}
//
//// visualize baseline setting
//
//if (minSignalSize > maxSignalSize) {
//	std::cout << "Max signal size must be greater or equal min signal size!" << std::endl;
//	exit(0);
//}
//
//// Reading gui setting
//bool gui;
//string guiString = cmd_line.get_arg("-gui");
//if (guiString != "") {
//	gui = to_bool(guiString);
//}
//else {
//	gui = true;
//}
//
//// Reading log setting
//bool log;
//string logString = cmd_line.get_arg("-log");
//if (logString != "") {
//	log = to_bool(logString);
//}
//else {
//	log = false;
//}
//
//// Reading downsample setting
//int downsample;
//string downsampleString = cmd_line.get_arg("-ds");
//if (downsampleString != "") {
//	downsample = atof(downsampleString.c_str());
//}
//else {
//	downsample = DEFAULT_DOWNSAMPLE;
//}
//
//
//
//std::ifstream test2(DNN_PROTO_PATH);
//if (!test2) {
//	std::cout << "DNN proto file not found!" << std::endl;
//	exit(0);
//}
//
//std::ifstream test3(DNN_MODEL_PATH);
//if (!test3) {
//	std::cout << "DNN model file not found!" << std::endl;
//	exit(0);
//}
//bool offlineMode = input != "";



void hrClient::processonvideo(std::string videoName, int samplingFrequency, int minSignalSize, int maxSignalSize, int chunkSize, bool gui) {
	//打开视频
	cv::VideoCapture cap;
	cap.open(videoName);
	if (!cap.isOpened()) {
		return;
	}

	std::string title = "rPPG offline";
	cout << title << endl;
	cout << "Processing " << videoName << endl;

	// Configure logfile path

	// Load video information
	const int WIDTH = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	const int HEIGHT = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	const double FPS = cap.get(cv::CAP_PROP_FPS);
	const double TIME_BASE = 0.001;

	// Print video information
	cout << "SIZE: " << WIDTH << "x" << HEIGHT << endl;
	cout << "FPS: " << FPS << endl;
	cout << "TIME BASE: " << TIME_BASE << endl;

	std::ostringstream window_title;
	window_title << title << " - " << WIDTH << "x" << HEIGHT << " -rppg  green";
	// Set up rPPG
	RPPG rppg = RPPG();
	rppg.load(
		WIDTH, HEIGHT, TIME_BASE, 1,
		samplingFrequency, 1,
		minSignalSize, maxSignalSize,
		chunkSize,
		DNN_PROTO_PATH, DNN_MODEL_PATH,
		gui);

	// Load landmark detector
	/*Ptr<Facemark> facemark = FacemarkLBF::create();
	facemark->loadModel("lbfmodel.yaml");*/

	// Load baseline if necessary

	cout << "HEART MEASURE!!" << endl;

	int i = 0;
	cv::Mat frameRGB = cv::Mat(WIDTH, HEIGHT, CV_16FC3), frameGray;


	while (true) {
		// Grab RGB frame
		cap.read(frameRGB);
		if (frameRGB.empty())
			break;
		// Generate grayframe
		cvtColor(frameRGB, frameGray, cv::COLOR_BGR2GRAY);
		equalizeHist(frameGray, frameGray);

		int time;
		time = (int)cap.get(cv::CAP_PROP_POS_MSEC);
		if (i % 1 == 0) {
			rppg.processFrame(frameRGB, frameGray, time);
		}
		else {
			cout << "SKIPPING FRAME TO DOWNSAMPLE!" << endl;
		}


		if (gui) {
			cv::imshow(window_title.str(), frameRGB);
			if (cv::waitKey(1) >= 0) break;
		}

		i++;
	}
	for (int i = 0; i < rppg.hrResult.size(); i++) {
		struct result r;
		struct result currentR = rppg.hrResult[i];
		r.length = currentR.length;
		for (int j = 0; j < currentR.length; j++) {
			r.reTrack[j] = currentR.reTrack[j];
			r.green[j] = currentR.green[j];
		}
		r.fps = currentR.fps;
		strcpy(r.state, currentR.state);
		hrResult.push_back(r);
	}

}
void hrClient::isFileExsistence()
{
	std::ifstream test1(DNN_PROTO_PATH);
	if (!test1) {
		std::cout << "DNN proto file not found!" << std::endl;
		exit(0);
	}

	std::ifstream test2(DNN_MODEL_PATH);
	if (!test2) {
		std::cout << "DNN model file not found!" << std::endl;
		exit(0);
	}

	std::ifstream test3(FACE_LANDMARK);
	if (!test3) {
		std::cout << "face_landmark file not found!" << std::endl;
		exit(0);
	}
}

int hrClient::processoncamera(int cameraindex, int sampletime, int samplingFrequency, int minSignalSize, int maxSignalSize, int chunkSize, bool gui)

{

	if (minSignalSize > maxSignalSize || minSignalSize < 4 || maxSignalSize>8) {
		std::cout << "Max signal size must be greater or equal min signal size!" << std::endl;
		exit(0);
	}
	cv::VideoCapture capture(cameraindex);


	//开线程获取视频
	std::deque<Measure<cv::Mat>>mat_deque;
	std::string videoname = "test1.mp4";
	int framenum = 0;
	int num = 0;
	// Configure logfile path
	const int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);//摄像头的宽
	const int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);//摄像头的高
	const double fps = capture.get(cv::CAP_PROP_FPS);//摄像头的fps
	const double TIME_BASE = 0.001;
	int  n = sampletime*fps;
	std::thread videomakethread(video_make, n, std::ref(num), std::ref(mat_deque), fps, width, height,capture);
	videomakethread.detach();
	// Print video information
	cout << "SIZE: " << width << "x" << height << endl;
	cout << "FPS: " << fps << endl;
	cout << "TIME BASE: " << TIME_BASE << endl;

	std::ostringstream window_title;
	window_title << " - " << width << "x" << height << " -rppg " ;

	// Set up rPPG
	RPPG rppg = RPPG();
	rppg.load(
		width, height, TIME_BASE, 1,
		samplingFrequency, 1,
		minSignalSize, maxSignalSize,
		chunkSize,
		DNN_PROTO_PATH, DNN_MODEL_PATH,
		 gui);
	// Load baseline if necessary

	cout << "START ALGORITHM" << endl;
	/*	int i = 0;*/
	cv::Mat frameRGB, frameGray;

	//每一帧处理
	while (framenum < n)
	{
		if (mat_deque.empty())
		{
			/*	Sleep(10);*/
			continue;
		}
		if (framenum >= num)
		{
			/*Sleep(10);*/
			continue;
		}
		else
		{
			frameRGB = mat_deque[framenum].val.clone();
			if (frameRGB.empty())
				break;
			cvtColor(frameRGB, frameGray, cv::COLOR_BGR2GRAY);
			equalizeHist(frameGray, frameGray);
			if (framenum % 1 == 0) {
				rppg.processFrame(frameRGB, frameGray, mat_deque[framenum].t);
			}
			else {
				cout << "SKIPPING FRAME TO DOWNSAMPLE!" << endl;
			}
			if (gui) {
				imshow(window_title.str(), frameRGB);
				if (cv::waitKey(1) >= 0) break;
			}
			++framenum;

		}

	}
	for (int i = 0; i < rppg.hrResult.size(); i++) {
		struct result r;
		struct result currentR = rppg.hrResult[i];
		r.length = currentR.length;
		for (int j = 0; j < currentR.length; j++) {
			r.reTrack[j] = currentR.reTrack[j];
			r.green[j] = currentR.green[j];
		}
		r.fps = currentR.fps;
		strcpy(r.state, currentR.state);
		hrResult.push_back(r);
	}


}


void video_make(int n, int& num,  std::deque<Measure<cv::Mat>> & data, const double fps, const int width, const int height,cv::VideoCapture capture)
{
	//cv::VideoWriter writer;
	TimerTimestamp capturetime = 0;
	//writer.open(videoname, cv::CAP_OPENCV_MJPEG, fps, cv::Size(width, height));
	while (capture.isOpened() && n > 0)
	{
		cv::Mat frame;
		capture >> frame;
		/*capturetime = (cv::getTickCount() * 1000.0) / cv::getTickFrequency();*/
		capturetime = int((num + 1) * 1000 / 30);
		if (frame.empty())
		{
			break;
		}
		/*writer << frame;*/
		n--;
		data.push_back(Measure<cv::Mat>(capturetime, frame));
		++num;
	}
}

