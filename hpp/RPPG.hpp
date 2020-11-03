
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
	char state[4];
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
		const int chunkSize, 
		const string dnnProtoPath, const string dnnModelPath,
		 const bool gui);

	void processFrame(cv::Mat frameRGB, cv::Mat frameGray, int time);
	//void processFrame_warp(cv::Mat& frameRGB, cv::Mat& frameGray, int time);

	void exit();
	typedef std::vector<cv::Point2f> Contour2f;
	std::deque<result> hrResult;



private:

	void detectFace(cv::Mat frameRGB, cv::Mat frameGray);
	void setNearestBox(std::vector<cv::Rect> boxes);
	void detectCorners(cv::Mat frameGray);
	void trackFace(cv::Mat frameGray);
	//void updateMask(cv::Mat& frameGray);
	//void updateROI();
	void extractSignal_g();
	//void extractSignal_pca();
	//void extractSignal_xminay();
	//void estimateHeartrate();
	//void draw(cv::Mat& frameRGB);
	void invalidateFace();
	//void log();
	void getFacePoints(std::vector<cv::Point> landMark);
   void findTriMax(cv::Mat powerSpectrum);
	//void findMaxInd(int start,int end);
	double findDistance(cv::Mat powerSpectrum);
	cv::Mat HSV_detector(cv::Mat src);
	void makeframedata(cv::Mat frameRGB, std::vector<cv::Point>ptr, std::vector<cv::Mat>& processframe);
	double getFps(cv::Mat& t, const double timeBase);
	void push(cv::Mat& m);

	// The algorithm
	rPPGAlgorithm rPPGAlg;
	// The classifier
	faceDetAlgorithm faceDetAlg;
	cv::CascadeClassifier haarClassifier;
	cv::dnn::Net dnnClassifier;

	// Settings
	cv::Size minFaceSize;
	int maxSignalSize;
	int minSignalSize;
	double rescanFrequency;
	double samplingFrequency;
	double timeBase;
	bool logMode;
	bool guiMode;
	
	// State variables
	int64_t time;
	double fps;
	int high;
	int64_t lastSamplingTime;
	int64_t lastScanTime;
	int low;
	int64_t now;
	bool faceValid=false;
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


	// Logfiles
	std::ofstream logfile;
	std::ofstream logfileDetailed;
	std::string logfilepath;
	
	//warp
	bool isFirst = true;
	cv::Point2f affine_transform_keypoints_first[5], affine_transform_keypoints_other[5];
	cv::Mat faceother;
    cv::Mat facefirst;
	cv::Rect firstBox;
	std::vector<cv::Point2f> firstlandmark;
	
	
	cv::Mat trans_other_to_first;
	cv::Mat warpped_other_face;
	cv::Mat smallframe;

	//vote
	//double vote[VOTENUM];
	//double maxnum;
	//std::vector<int> maxind;
	int frameCount = 0;
	int inds[5];
	int chunkSize, chunkSizeCount=0;
};

#endif /* RPPG_hpp */