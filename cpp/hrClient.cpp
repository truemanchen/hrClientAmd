

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
#define DNN_MODEL_PATH "res10_300x300_ssd_iter_140000.caffemodel"//�������������ļ�
#define FACE_LANDMARK "shape_predictor_68_face_landmarks.dat"//���������һ���ļ�


void video_make(int n, int& num,  std::deque<Measure<cv::Mat>>& data, const double fps, const int width, const int height,cv::VideoCapture capture);





//RPPG�ദ����ǵ�֡ͼ�񣬲���¼�������źţ������źŴﵽһ�����ȵ�ʱ�� ���д����źż�¼��hrResult�ṹ����
//demo  ���ʹ��RPPG����һ���Ѿ��ĺõ���Ƶ
//������Ƶ  ����Ƶ��ÿһ֡����RPPG �������ɫͨ������  �����㹻���Ϳ��Դ��ݸ���������������Ԥ��
void hrClient::processonvideo(std::string videoName, int samplingFrequency, int minSignalSize, int maxSignalSize, int slideWindowStep, bool gui) {
	if (minSignalSize > maxSignalSize || minSignalSize < 4 || maxSignalSize>8) {
		std::cout << "Max signal size must be greater or equal min signal size!" << std::endl;
		exit(0);
	}

	//����Ƶ
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
	const int WIDTH = cap.get(cv::CAP_PROP_FRAME_WIDTH);//��ȡ��Ƶ�Ŀ�
	const int HEIGHT = cap.get(cv::CAP_PROP_FRAME_HEIGHT);//��ȡ��Ƶ�ĳ�
	const double FPS = cap.get(cv::CAP_PROP_FPS);//��ȡ��Ƶ��֡��  ��Ȼ֮������¼���
	const double TIME_BASE = 0.001;

	// ��ӡ��Ƶ�ľ�����Ϣ
	cout << "SIZE: " << WIDTH << "x" << HEIGHT << endl;
	cout << "FPS: " << FPS << endl;
	cout << "TIME BASE: " << TIME_BASE << endl;

	std::ostringstream window_title;
	window_title << title << " - " << WIDTH << "x" << HEIGHT << " -rppg  green";
	
	
	// //��ʼ��RPPG��Ĳ���
	//������ʼ��RPPG�Ĳ��� �ֱ���
    //��Ƶ֡�Ŀ�  ��Ƶ֡�ĸ�  ms��s�Ļ���  ����֡��ȡһ����ɫͨ���źţ���������
   //���ʵĸ���Ƶ�ʣ�10����0.1s����һ���Ѿ��󵽵����ʶ��У�  ���������¼��Ƶ��
  //��ɫͨ���źŵ���С����   ��ɫͨ���źŵ���󳤶� slideWindowStep�������֡����һ�η���(������˵ÿ����1֡  ��Ҫ��ȡһ�������źŶ���  �����е��˷�ʱ�� ����ÿchunksize��ȡһ��)
  //����ʶ��������ļ��ĵ�ַ   
  //�Ƿ񵼳���ɫͨ���ź�   �Ƿ���ӻ�
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
		// ��ȡ��Ƶ֡
		cap.read(frameRGB);

		//��Ƶ������ �ж�ѭ��
		if (frameRGB.empty())
			break;


		//ת���ɻҶ�ͼ ��������ʶ��
		cvtColor(frameRGB, frameGray, cv::COLOR_BGR2GRAY);
		
		//cout << "frameRGB type:"<<frameRGB.type() << endl;
		//cout << "framegray type: " << frameGray.type() << endl;
		//equalizeHist(frameRGB, frameRGB);

		//��ȡÿ֡��ʱ���
		int time;
		time = (int)cap.get(cv::CAP_PROP_POS_MSEC);
		if (i % 1 == 0) {
			rppg.processFrame(frameRGB, frameGray, time);//���Ĺ���  ��ȡ�ź�  ������
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

	//��RPPG��ȡ���ź�  ���Ƶ�hrClient��
	for (int i = 0; i < rppg.hrResult.size(); i++) {
		struct result r;
		struct result currentR = rppg.hrResult[i];//RPPG��ǰ���ź�
		r.length = currentR.length;//�����鳤�ȸ�������
		for (int j = 0; j < currentR.length; j++) {//������������źŸ�������
			r.reTrack[j] = currentR.reTrack[j];
			r.green[j] = currentR.green[j];
			r.timeStamp[j] = currentR.timeStamp[j];
		}
		r.fps = currentR.fps;//�ѵ�ǰ֡��FPS��������

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


//��������ͷ
int hrClient::processoncamera(int cameraindex, int sampletime, int samplingFrequency, int minSignalSize, int maxSignalSize, int slideWindowStep, bool gui)

{

	if (minSignalSize > maxSignalSize || minSignalSize < 4 || maxSignalSize>8) {
		std::cout << "Max signal size must be greater or equal min signal size!" << std::endl;
		exit(0);
	}
	cv::VideoCapture capture(cameraindex);


	//���̻߳�ȡ��Ƶ
	std::deque<Measure<cv::Mat>>mat_deque;
	std::string videoname = "test1.mp4";
	int framenum = 0;
	int num = 0;
	// ��ȡ����ͷ�ĸ�������
	const int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);//����ͷ�Ŀ�
	const int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);//����ͷ�ĸ�
	const double fps = capture.get(cv::CAP_PROP_FPS);//����ͷ��fps
	const double TIME_BASE = 0.001;
	int  n = sampletime*fps;//�õ���֡��

	//����һ���߳�¼����Ƶ  �����ֱ��� ¼�ƺ��� ��ʣ����֡δ��¼  ��¼�˶���֡  ֡�Ĵ��λ�� ����ͷ����
	std::thread videomakethread(video_make, n, std::ref(num), std::ref(mat_deque), fps, width, height,capture);
	videomakethread.detach();//ɱ���߳�
	// ��ӡ����ͷ�ĸ�����Ϣ
	cout << "SIZE: " << width << "x" << height << endl;
	cout << "FPS: " << fps << endl;
	cout << "TIME BASE: " << TIME_BASE << endl;

	std::ostringstream window_title;
	window_title << " - " << width << "x" << height << " -rppg " ;

	// // ��ʼ��RPPG��Ĳ���
		//������ʼ��RPPG�Ĳ��� �ֱ���
		//��Ƶ֡�Ŀ�  ��Ƶ֡�ĸ�  ms��s�Ļ���  ����֡��ȡһ����ɫͨ���źţ���������
	   //���ʵĸ���Ƶ�ʣ�10����0.1s����һ���Ѿ��󵽵����ʶ��У�  ���������¼��Ƶ��
	  //��ɫͨ���źŵ���С����   ��ɫͨ���źŵ���󳤶�
	  //����ʶ��������ļ��ĵ�ַ   slideWindowStep�������֡����һ�η���
	  //�Ƿ񵼳���ɫͨ���ź�   �Ƿ���ӻ�
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

	//ÿһ֡����
	while (framenum < n)
	{
		if (mat_deque.empty())//û��Ҫ�����֡ ��Ҫ��һ��
		{
			/*	Sleep(10);*/
			continue;
		}
		if (framenum >= num)//����֡�Ѿ������˼�¼��֡ ��Ҫ��һ��
		{
			/*Sleep(10);*/
			continue;
		}
		else//��ʼ����
		{
			frameRGB = mat_deque[framenum].val.clone();
			if (frameRGB.empty())
				break;
			cvtColor(frameRGB, frameGray, cv::COLOR_BGR2GRAY);
			equalizeHist(frameGray, frameGray);//ֱ��ͼ���⻯
			if (framenum % 1 == 0) {
				rppg.processFrame(frameRGB, frameGray, mat_deque[framenum].t);//����֡  ����������Ϣ��¼����
			}
			else {
				cout << "SKIPPING FRAME TO DOWNSAMPLE!" << endl;
			}
			if (1) {
				imshow(window_title.str(), frameRGB);
				if (cv::waitKey(1) >= 0) break;
			}
			++framenum;//�Ѿ������֡��

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
	}//��RPPG���źż�¼���ͻ�������


}


void video_make(int n, int& num,  std::deque<Measure<cv::Mat>> & data, const double fps, const int width, const int height,cv::VideoCapture capture)
{
	//cv::VideoWriter writer;
	TimerTimestamp capturetime = 0;//ÿһ֡��ʱ���
	//writer.open(videoname, cv::CAP_OPENCV_MJPEG, fps, cv::Size(width, height));
	while (capture.isOpened() && n > 0)//��¼��֡����Ϊ0
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
		data.push_back(Measure<cv::Mat>(capturetime, frame));//ÿһ֡�������
		++num;
	}
}

