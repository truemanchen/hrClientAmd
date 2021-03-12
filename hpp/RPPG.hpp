
#ifndef RPPG_hpp
#define RPPG_hpp

#include <fstream>
#include <string>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include<deque>
#include <stdio.h>

//using namespace cv;
//using namespace cv::dnn;
using namespace std;


enum rPPGAlgorithm { g, pca, xminay };
enum faceDetAlgorithm { haar, deep };
struct result {
	double green[1000];
	int reTrack[1000];
	int length;
	double fps;
	int timeStamp[1000];
};

class RPPG {

public:

	// Constructor
	RPPG() { ; }

	// Load Settings
	bool load(
		const int width, const int height, const double timeBase, const int downsample,
		const double samplingFrequency, const double rescanFrequency,
		const int minSignalSize, const int maxSignalSize,
		const int slideWindowStep, 
		const string dnnProtoPath, const string dnnModelPath,
		 const bool gui);

	void processFrame(const cv::Mat& frameRGB, const cv::Mat& frameGray, const int time);
	//void processFrame_warp(cv::Mat& frameRGB, cv::Mat& frameGray, int time);

	typedef std::vector<cv::Point2f> Contour2f;
	void invalidateFace();
	std::deque<result> hrResult;
	bool isFaceValid();
	void brightnessException(cv::Mat InputImg, float& cast, float& da);



private:
	void histOfArea(cv::Mat face, cv::Mat faceMask);
	void drawHist(vector<int> nums);

	void detectFace(const cv::Mat& frameRGB, const cv::Mat& frameGray);
	void setNearestBox(std::vector<cv::Rect> boxes);
	void detectCorners(const cv::Mat& frameGray);
	void trackFace(const cv::Mat& frameGray);
	//void updateMask(cv::Mat& frameGray);
	//void updateROI();
	void extractSignal_g();
	//void extractSignal_pca();
	//void extractSignal_xminay();
	//void estimateHeartrate();
	//void draw(cv::Mat& frameRGB);
	//void log();
	cv::Mat HSV_detector(const cv::Mat& src);
	void makeframedata(const cv::Mat& frameRGB, std::vector<cv::Point>ptr, std::vector<cv::Mat>& processframe);
	double getFps(cv::Mat& t, const double timeBase);
	void push(cv::Mat& m);

	// 心率识别算法
	rPPGAlgorithm rPPGAlg;
	// 人脸识别算法
	faceDetAlgorithm faceDetAlg;
	cv::CascadeClassifier haarClassifier;
	cv::dnn::Net dnnClassifier;

	// 设置
	cv::Size minFaceSize;
	int maxSignalSize;
	int minSignalSize;
	double rescanFrequency;
	double samplingFrequency;
	double timeBase;
	bool guiMode;
	
	// State variables
	int64_t time;
	double fps;
	int high;
	int64_t lastSamplingTime;
	int64_t lastScanTime;
	int low;
	int64_t now;
	bool faceValid;
	bool rescanFlag;
	

	// Tracking
	cv::Mat lastFrameGray;
	Contour2f corners;
	cv::Ptr<cv::face::Facemark> facemark;
	

	// Mask
	cv::Rect box;
	cv::Mat1b mask;
	/*cv::Rect roi;
	cv::Rect roi2;*/

	// Raw signal
	cv::Mat1d s/*,s1,s2,s3,s4*/;
	cv::Mat1d t;
	cv::Mat1b re;
	cv::Mat1d allFrameNumber;


	
	//warp
	//cv::Point2f affine_transform_keypoints_first[5], affine_transform_keypoints_other[5];
	//cv::Mat faceother;
 //   cv::Mat facefirst;
	//cv::Rect firstBox;
	//std::vector<cv::Point2f> firstlandmark;
	//
	//
	//cv::Mat trans_other_to_first;
	//cv::Mat warpped_other_face;
	cv::Mat smallframe;

	//vote
	//double vote[VOTENUM];
	//double maxnum;
	//std::vector<int> maxind;
	int frameCount ;
	int slideWindowStep, slideWindowStepCount=0;


	//状态码
	//bool state[5];//第二位代表有没有人脸 第三位代表能不能是否有重新追踪 第四位代表有没有足够的心率信号长度 0是没有 1是有

	//结果

};

#endif /* RPPG_hpp */