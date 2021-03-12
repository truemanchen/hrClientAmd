
#ifndef hrServer_hpp
#define hrServer_hpp

#include <fstream>
#include <string>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <stdio.h>

//using namespace cv;
//using namespace cv::dnn;
using namespace std;

class hrServer {

public:
	hrServer(){
		meanBpm = 0;
	}
	~hrServer(){}
	void extractSignal_g(int reTrack[], double green[], int length, double fps,int timeStamps[]);
	//void extractSignal_pca();
	//void extractSignal_xminay();
	double estimateHeartrate();



private:

	void findTriMax(cv::Mat powerSpectrum);
	//void findMaxInd(int start,int end);
	void normalization(cv::InputArray _a, cv::OutputArray _b);
	void denoise(cv::InputArray _a, cv::InputArray _jumps, cv::OutputArray _b);
	void detrend(cv::InputArray _a, cv::OutputArray _b, int lambda);
	void movingAverage(cv::InputArray _a, cv::OutputArray _b, int n, int s);
	void timeToFrequency(cv::InputArray _a, cv::OutputArray _b, bool magnitude);
	void array2Mat(double green[], int length);
	void array2Mat(int reTrack[], int length);
	void extractHeartrateVariability();
	void findPeaks(double* src, int src_lenth, double distance, int* indMax, int* indMax_len);
	double* ideal_bandpass_filter(double* input, int length, float fl, float fh, float fps);



	// Settings
	//cv::Size minFaceSize;
	//int maxSignalSize;
	//int minSignalSize;
	//double rescanFrequency;
	//double samplingFrequency;
	//double timeBase;
	//bool logMode;
	//bool guiMode;

	// State variables
	//int64_t time;
	double fps;
	int high;
	//int64_t lastSamplingTime;
	int low;
	int64_t now;
	//bool faceValid = false;
	//bool rescanFlag;


	//// Tracking
	//cv::Mat lastFrameGray;
	//cv::Ptr<cv::face::Facemark> facemark;


	// Mask
	//cv::Rect box;
	//cv::Mat1b mask;
	/*cv::Rect roi;
	cv::Rect roi2;*/

	// Raw signal
	cv::Mat1d s/*,s1,s2,s3,s4*/;
	cv::Mat1b re;

	// Estimation
	cv::Mat1d s_f/*, s_f1, s_f2, s_f3, s_f4*/;
	cv::Mat1d bpms, bpmsc,snrs,hrvs/*, bpms1, bpms2, bpms3, bpms4,idealbpms*/;

	cv::Mat1d powerSpectrum/*, powerSpectrum1, powerSpectrum2, powerSpectrum3, powerSpectrum4*/;
	double bpm /*, bpm1 = 0.0, bpm2 = 0.0, bpm3 = 0.0, bpm4 = 0.0,bpm5=0.0,idealbpm=0.0*/;
	int isbpmgood ;
	double lastbpm,lasthrv,lastsnr;
	double meanBpm,meanHrv,meanSnr ;
	double prominentFrequency;
	double unitFrequency,snr,hrv;
	// Logfiles
	//std::ofstream logfile;
	//std::ofstream logfileDetailed;
	//std::string logfilepath;

	//warp
	//bool isFirst = true;
	//cv::Point2f affine_transform_keypoints_first[5], affine_transform_keypoints_other[5];
	//cv::Mat faceother;
	//cv::Mat facefirst;
	//cv::Rect firstBox;
	//std::vector<cv::Point2f> firstlandmark;


	//cv::Mat trans_other_to_first;
	//cv::Mat warpped_other_face;
	//cv::Mat smallframe;

	//vote
	//double vote[VOTENUM];
	//double maxnum;
	//std::vector<int> maxind;
	int frameCount = 0;
	int inds[5];


};

#endif /* hrServer_hpp */