

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
#define DEFAULT_RESCAN_FREQUENCY 0.25
#define DEFAULT_SAMPLING_FREQUENCY 1
#define DEFAULT_MIN_SIGNAL_SIZE 5
#define DEFAULT_MAX_SIGNAL_SIZE 7
#define DEFAULT_FRAMES          1200
#define DEFAULT_DOWNSAMPLE 1// x means only every xth frame is used
#define WARP 0

#define DNN_PROTO_PATH "deploy.prototxt"
#define DNN_MODEL_PATH "res10_300x300_ssd_iter_140000.caffemodel"//人脸检测的两个文件
#define FACE_LANDMARK "shape_predictor_68_face_landmarks.dat"//特征点检测的一个文件


void video_make(int n, int& num,  std::deque<Measure<cv::Mat>>& data, const double fps, const int width, const int height,cv::VideoCapture capture);





//RPPG类处理的是单帧图像，并记录处理后的信号，当该信号达到一定长度的时候 进行处理，信号记录在hrResult结构体中
//demo  如何使用RPPG处理一个已经拍好的视频
//处理视频  把视频的每一帧交给RPPG 并获得颜色通道序列  序列足够长就可以传递给服务器进行心率预测
void hrClient::processonvideo(std::string videoName, int samplingFrequency, int minSignalSize, int maxSignalSize, int slideWindowStep, bool gui) {
	if (minSignalSize > maxSignalSize || minSignalSize < 4 || maxSignalSize>8) {
		std::cout << "Max signal size must be greater or equal min signal size!" << std::endl;
		exit(0);
	}

	//打开视频
	cv::VideoCapture cap;
	cap.open(videoName,cv::CAP_FFMPEG);
	if (!cap.isOpened()) {
		cout << "the video doesn't exsis!" << endl;
		return;
	}

	std::string title = "rPPG video";
	cout << title << endl;
	cout << "Processing " << videoName << endl;

	// Configure logfile path

	// Load video information
	const int WIDTH = cap.get(cv::CAP_PROP_FRAME_WIDTH);//获取视频的宽
	const int HEIGHT = cap.get(cv::CAP_PROP_FRAME_HEIGHT);//获取视频的长
	const double FPS = cap.get(cv::CAP_PROP_FPS);//获取视频在帧率  虽然之后会重新计算
	const double TIME_BASE = 0.001;

	// 打印视频的具体信息
	cout << "SIZE: " << WIDTH << "x" << HEIGHT << endl;
	cout << "FPS: " << FPS << endl;
	cout << "TIME BASE: " << TIME_BASE << endl;

	std::ostringstream window_title;
	window_title << title << " - " << WIDTH << "x" << HEIGHT << " -rppg  green";
	
	
	// //初始化RPPG类的参数
	//用来初始化RPPG的参数 分别是
    //视频帧的宽  视频帧的高  ms与s的换算  多少帧提取一次颜色通道信号（降采样）
   //心率的更新频率（10代表0.1s重置一次已经求到的心率队列）  人脸的重新检测频率
  //颜色通道信号的最小长度   颜色通道信号的最大长度 slideWindowStep代表多少帧进行一次分析(按理来说每加入1帧  就要提取一次心率信号队列  但是有点浪费时间 可以每chunksize提取一次)
  //人脸识别的两个文件的地址   
  //是否导出颜色通道信号   是否可视化
	RPPG rppg =  RPPG();
	rppg.load(
		WIDTH, HEIGHT, TIME_BASE, 1,
		samplingFrequency, DEFAULT_RESCAN_FREQUENCY,
		minSignalSize, maxSignalSize,
		slideWindowStep,
		DNN_PROTO_PATH, DNN_MODEL_PATH,
		gui);


	// Load baseline if necessary

	cout << "HEART MEASURE!!" << endl;

	int i = 0;
	cv::Mat frameRGB = cv::Mat(WIDTH, HEIGHT, CV_16FC3), frameGray;


	while (true) {
		// 读取视频帧
		cap.read(frameRGB);

		//视频结束后 中断循环
		if (frameRGB.empty())
			break;


		//转换成灰度图 用来人脸识别
		cvtColor(frameRGB, frameGray, cv::COLOR_BGR2GRAY);
		
		//cout << "frameRGB type:"<<frameRGB.type() << endl;
		//cout << "framegray type: " << frameGray.type() << endl;
		//equalizeHist(frameRGB, frameRGB);

		//获取每帧的时间戳
		int time;
		time = (int)cap.get(cv::CAP_PROP_POS_MSEC);
		if (i % 1 == 0) {
			rppg.processFrame(frameRGB, frameGray, time);//核心功能  提取信号  并保存
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

	//把RPPG提取的信号  复制到hrClient中
	for (int i = 0; i < rppg.hrResult.size(); i++) {
		struct result r;
		struct result currentR = rppg.hrResult[i];//RPPG当前的信号
		r.length = currentR.length;//把数组长度复制下来
		for (int j = 0; j < currentR.length; j++) {//把两个数组的信号复制下来
			r.reTrack[j] = currentR.reTrack[j];
			r.green[j] = currentR.green[j];
			r.timeStamp[j] = currentR.timeStamp[j];
		}
		r.fps = currentR.fps;//把当前帧的FPS复制下来

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


//处理摄像头
int hrClient::processoncamera(int cameraindex, int sampletime, int samplingFrequency, int minSignalSize, int maxSignalSize, int slideWindowStep, bool gui)

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
	// 提取摄像头的各个属性
	const int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);//摄像头的宽
	const int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);//摄像头的高
	const double fps = capture.get(cv::CAP_PROP_FPS);//摄像头的fps
	const double TIME_BASE = 0.001;
	int  n = sampletime*fps;//得到总帧数

	//开辟一个线程录制视频  参数分别是 录制函数 还剩多少帧未记录  记录了多少帧  帧的存放位置 摄像头参数
	std::thread videomakethread(video_make, n, std::ref(num), std::ref(mat_deque), fps, width, height,capture);
	videomakethread.detach();//杀死线程
	// 打印摄像头的各种信息
	cout << "SIZE: " << width << "x" << height << endl;
	cout << "FPS: " << fps << endl;
	cout << "TIME BASE: " << TIME_BASE << endl;

	std::ostringstream window_title;
	window_title << " - " << width << "x" << height << " -rppg " ;

	// // 初始化RPPG类的参数
		//用来初始化RPPG的参数 分别是
		//视频帧的宽  视频帧的高  ms与s的换算  多少帧提取一次颜色通道信号（降采样）
	   //心率的更新频率（10代表0.1s重置一次已经求到的心率队列）  人脸的重新检测频率
	  //颜色通道信号的最小长度   颜色通道信号的最大长度
	  //人脸识别的两个文件的地址   slideWindowStep代表多少帧进行一次分析
	  //是否导出颜色通道信号   是否可视化
	RPPG rppg = RPPG();
	rppg.load(
		width, height, TIME_BASE, 1,
		samplingFrequency, 1,
		minSignalSize, maxSignalSize,
		slideWindowStep,
		DNN_PROTO_PATH, DNN_MODEL_PATH,
		 gui);
	// Load baseline if necessary

	cout << "START ALGORITHM" << endl;
	/*	int i = 0;*/
	cv::Mat frameRGB, frameGray;

	//每一帧处理
	while (framenum < n)
	{
		if (mat_deque.empty())//没有要处理的帧 需要等一等
		{
			/*	Sleep(10);*/
			continue;
		}
		if (framenum >= num)//处理帧已经赶上了记录的帧 需要等一等
		{
			/*Sleep(10);*/
			continue;
		}
		else//开始处理
		{
			frameRGB = mat_deque[framenum].val.clone();
			if (frameRGB.empty())
				break;
			cvtColor(frameRGB, frameGray, cv::COLOR_BGR2GRAY);
			equalizeHist(frameGray, frameGray);//直方图均衡化
			if (framenum % 1 == 0) {
				rppg.processFrame(frameRGB, frameGray, mat_deque[framenum].t);//处理帧  并把心率信息记录下来
			}
			else {
				cout << "SKIPPING FRAME TO DOWNSAMPLE!" << endl;
			}
			if (1) {
				imshow(window_title.str(), frameRGB);
				if (cv::waitKey(1) >= 0) break;
			}
			++framenum;//已经处理的帧数

		}

	}
	for (int i = 0; i < rppg.hrResult.size(); i++) {
		struct result r;
		struct result currentR = rppg.hrResult[i];
		r.length = currentR.length;
		for (int j = 0; j < currentR.length; j++) {
			r.reTrack[j] = currentR.reTrack[j];
			r.green[j] = currentR.green[j];
			r.timeStamp[j] = currentR.timeStamp[j];
		}
		r.fps = currentR.fps;
		hrResult.push_back(r);
	}//把RPPG的信号记录到客户端类中


}


void video_make(int n, int& num,  std::deque<Measure<cv::Mat>> & data, const double fps, const int width, const int height,cv::VideoCapture capture)
{
	//cv::VideoWriter writer;
	TimerTimestamp capturetime = 0;//每一帧的时间戳
	//writer.open(videoname, cv::CAP_OPENCV_MJPEG, fps, cv::Size(width, height));
	while (capture.isOpened() && n > 0)//待录制帧数不为0
	{
		cv::Mat frame;
		capture >> frame;
		/*capturetime = (cv::getTickCount() * 1000.0) / cv::getTickFrequency();*/
		capturetime = int((num + 1) * 1000 / fps);
		if (frame.empty())
		{
			break;
		}
		/*writer << frame;*/
		n--;
		data.push_back(Measure<cv::Mat>(capturetime, frame));//每一帧存放下来
		++num;
	}
}

