
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


#define LOW_BPM 42//人类心率最低
#define HIGH_BPM 180//人类心率最高
#define REL_MIN_FACE_SIZE 0.4//人脸追踪的时候 最小人脸尺寸比例
#define SEC_PER_MIN 60//计算一分钟内人的心率
#define MAX_CORNERS 5//最大角点数
#define MIN_CORNERS 3//最小角点数
#define QUALITY_LEVEL 0.01//角点的提取质量  不用改
#define MIN_DISTANCE 25
#define MAXNUMBERS 3//提取功率谱中能量最大的频率的数量
dlib::shape_predictor sp;
dlib::full_object_detection shape;//特征点提取后存放的位置
string modelpath = "shape_predictor_68_face_landmarks.dat";


//用来初始化RPPG的参数 分别是
//视频帧的宽  视频帧的高  ms与s的换算  多少帧提取一次颜色通道信号（降采样）
//心率的更新频率（10代表0.1s重置一次已经求到的心率队列）  人脸的重新检测频率
//颜色通道信号的最小长度   颜色通道信号的最大长度
//人脸识别的两个文件的地址   slideWindowStep代表多少帧进行一次信号提取
//是否导出颜色通道信号   是否可视化
bool RPPG::load(
	const int width, const int height, const double timeBase, const int downsample,
	const double samplingFrequency, const double rescanFrequency,
	const int minSignalSize, const int maxSignalSize,
	const int slideWindowStep,
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
	this->slideWindowStep = slideWindowStep;
	this->frameCount = 0;
	this->faceValid = false;
	// 加载人脸检测模型

	dnnClassifier = readNetFromCaffe(dnnProtoPath, dnnModelPath);

	//加载特征点模型
	dlib::deserialize(modelpath) >> sp;
	frameCount = 0;

	return true;
}

//利用hsv空间进行皮肤检测
cv::Mat RPPG::HSV_detector(const cv::Mat& src)
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
			if (p_src[h] >= 3 && p_src[h] <= 25 && p_src[s] >= 20 && p_src[s] <= 220 && p_src[v] >= 120)

			{
				p_mask[0] = 255;
			}
		}
	}
	cv::Mat detect;
	src.copyTo(detect, output_mask);;
	return output_mask;
}

//计算每个颜色队列的fps即采样率
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
	//if (!t.empty()) {
	//	//cout << "t 和 索引一样多吗" << (t.rows == allFrameNumber.rows) << endl;
	//	for (int i = 0; i < t.rows; i++) {
	//		cout << t.at<int>(i, 0);
	//		i != (t.rows - 1) ? cout << " " : cout << endl;
	//	}
	//	for (int i = 0; i < allFrameNumber.rows; i++) {
	//		cout << allFrameNumber.at<int>(i, 0);
	//		i != (allFrameNumber.rows - 1) ? cout << " " : cout << endl;
	//	}

	//}
	return result;
}

//先进先出 加入新帧 淘汰最早进入的帧
void RPPG::push(cv::Mat& m) {
	const int length = m.rows;
	m.rowRange(1, length).copyTo(m.rowRange(0, length - 1));
	m.pop_back();
}


//返回人脸是否有效
bool RPPG::isFaceValid() {
	return faceValid;
}

//处理视频帧
void RPPG::processFrame(const cv::Mat& frameRGB, const cv::Mat& frameGray, const int time) {
	cv::Mat skinMask[2];
	std::vector<cv::Mat> sMs;//存放不同人脸区域的皮肤mask
	std::vector<cv::Mat> masks;

	this->time = time;
	if (!faceValid) {//没有成功检测到人脸
		cout << "Not valid, finding a new face" << endl;

		lastScanTime = time;
		detectFace(frameRGB, frameGray);//进行人脸检测

	}
	//已经进行了人脸检测，后续的是采用追踪的方式进行人脸检测，为了准确，再次进行人脸检测
	else if ((time - lastScanTime) * timeBase >= 1 / rescanFrequency) {

		cout << "Valid, but rescanning face" << endl;

		lastScanTime = time;//记录上次检测的时间 方便隔一段时间再次进行检测

		detectFace(frameRGB, frameGray);
		rescanFlag = true;

	}
	else {

		cout << "Tracking face" << endl;////追踪人脸不检测，减少时间耗费

		trackFace(frameGray);


	}

	if (faceValid && (!box.empty())) {//检测到人脸
		std::vector<cv::Point> landMark;//特征点的opencv变量
		//对人脸检测到的矩形框裁剪  防止越界到整个视频帧外面去 取一个交集
		box = box & cv::Rect(0, 0, frameRGB.cols, frameRGB.rows);
		//把人脸从视频帧中抠下来
		frameRGB(box).copyTo(smallframe);
		//将人脸从opencv格式转化成dlib格式
		dlib::cv_image<dlib::rgb_pixel> dlib_frame(smallframe);
		//将人脸矩形框从opencv格式转化成dlib格式
		dlib::rectangle dlibRect(0, 0,
			0 + smallframe.cols, 0 + smallframe.rows);
		//特征点检测 得到dlib格式的特征点
		shape = sp(dlib_frame, dlibRect);
		//将dlib格式的特征点转化成opencv格式
		for (int i = 0; i < 68; i++) {
			cv::Point temp = cv::Point(shape.part(i).x(), shape.part(i).y());
			landMark.push_back(temp);
		}

		//对整张脸进行皮肤检测
		cv::Rect smallbox = cv::Rect(cv::Point(max(0, box.tl().x), max(0, box.tl().y)),
			cv::Point(box.tl().x + box.width, box.tl().y + box.height));
		skinMask[0] = HSV_detector(frameRGB(smallbox));


		//将整张脸的皮肤检测划分成不同的区域
		makeframedata(skinMask[0], landMark, sMs);

		// 更新fps
		//fps = getFps(t, timeBase);

		// 当信号队列超过最大值  移除最早的旧信号  加入新信号
		//移除信号的时间戳  加入新信号的时间戳
		while (s.rows > fps * maxSignalSize) {
			push(s);
			/*push(s1);
			push(s2);
			push(s3);
			push(s4);*/
			push(t);
			push(re);
			push(allFrameNumber);
		}
		assert(s.rows == t.rows && s.rows == re.rows);//判断信号是否完整

		// 求人脸所选区域三通道的平均值  按照RGB的顺序 0是全部
		cv::Scalar means = mean(smallframe, skinMask[0].empty() ? cv::noArray() : sMs[0]);
		cv::imshow("sm", sMs[0]);
		histOfArea(smallframe, sMs[0]);

		// 放入double数组
		double values[] = { means(0), means(1), means(2) };


		//颜色信号反映了心率信号
		s.push_back(cv::Mat(1, 3, CV_64F, values));
		/*s1.push_back(cv::Mat(1, 3, CV_64F, values1));
		s2.push_back(cv::Mat(1, 3, CV_64F, values2));
		s3.push_back(cv::Mat(1, 3, CV_64F, values3));
		s4.push_back(cv::Mat(1, 3, CV_64F, values4));*/
		t.push_back(time);
		allFrameNumber.push_back(frameCount);
		// 保存是否重新进行人脸检测的标志
		re.push_back(rescanFlag);

		// 更新fps
		fps = getFps(t, timeBase);

		// 人体心率的正常范围对应的功率谱位置
		low = (int)(s.rows * LOW_BPM / SEC_PER_MIN / fps);
		high = (int)(s.rows * HIGH_BPM / SEC_PER_MIN / fps) + 1;


		// 如果信号足够长进行分析，并且恰好是需要分析的那帧  把颜色队列提取出来
			//此时可以交给服务器分析  颜色队列放入mat中  记录此时的fps
		if (s.rows >= fps * minSignalSize && slideWindowStepCount % slideWindowStep == 0) {
			//state[3] = true;
			struct result r;
			r.length = s.rows;
			for (int i = 0; i < s.rows; i++) {
				r.green[i] = s.at<double>(i, 1);
				r.reTrack[i] = re.at<uchar>(i, 0);
				r.timeStamp[i] = t.at<int>(i, 0);
			}
			//for (int i = 0; i < 5; i++) {
			//	r.state[i] = state[i];
			//}
			r.fps = fps;
			hrResult.push_back(r);

		}
		//else if(s.rows < fps)
		//{
		//	state[3] = false;
		//}
		rescanFlag = false;

		frameGray.copyTo(lastFrameGray);
		//state[1] = faceValid;
		slideWindowStepCount ++;
		/*boxroi.pop_back();*/
	}
	frameCount++;
}

//人脸检测的函数
void RPPG::detectFace(const cv::Mat& frameRGB, const cv::Mat& frameGray) {

	std::cout << "scanning for faces" << endl;
	std::vector<cv::Rect> boxes = {};
	switch (faceDetAlg) {
	case haar:
		// haar特征进行人脸检测
		haarClassifier.detectMultiScale(frameGray, boxes, 1.1, 2, cv::CASCADE_SCALE_IMAGE, minFaceSize);
		break;
	case deep:
		// 深度神经网络检测人脸
		cv::Mat resize300;
		cv::resize(frameRGB, resize300, cv::Size(300, 300));
		//进行人脸检测前的预处理
		cv::Mat blob = blobFromImage(resize300, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));
		dnnClassifier.setInput(blob);
		cv::Mat detection = dnnClassifier.forward();
		cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
		float confidenceThreshold = 0.6;
		//达到阈值后的，记录下人脸矩形框的位置
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

		//得到最小的人脸矩形框
		setNearestBox(boxes);
		//进行角点检测
		detectCorners(frameGray);
		//updateROI();
		//updateMask(frameGray);
		faceValid = true;

		//state[1] = true;
	}
	else {
		//没有检测到人脸  重置信号序列
		cout << "Found no face" << endl;
		//state[1] = false;
		invalidateFace();
	}
}


//得到最小的人脸矩形框
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
//进行角点检测
void RPPG::detectCorners(const cv::Mat& frameGray) {

	// 定义追踪区域
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
	fillPoly(trackingRegion, pts, npts, 1, cv::Scalar(255, 255, 255));//将追踪区域设置为掩膜

	// 在全脸进行角点检测
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


//利用角点追踪人脸
void RPPG::trackFace(const cv::Mat& frameGray) {

	// 角点不够  重新进行角点检测
	if (corners.size() < MIN_CORNERS) {
		detectCorners(frameGray);
	}

	Contour2f corners_1;
	Contour2f corners_0;
	std::vector<uchar> cornersFound_1;
	std::vector<uchar> cornersFound_0;
	cv::Mat err;

	// 利用光流追踪法，在下一帧中把上一帧的角点的位置追踪到
	calcOpticalFlowPyrLK(lastFrameGray, frameGray, corners, corners_1, cornersFound_1, err);

	// 反追踪回去
	calcOpticalFlowPyrLK(frameGray, lastFrameGray, corners_1, corners_0, cornersFound_0, err);

	// 反追踪后的角点如果与原角点位置小于2像素 就是鲁棒的角点  并且保存下来
	Contour2f corners_1v;
	Contour2f corners_0v;
	for (size_t j = 0; j < corners.size(); j++) {
		if (cornersFound_1[j] && cornersFound_0[j]
			&& norm(corners[j] - corners_0[j]) < 2) {
			corners_0v.push_back(corners_0[j]);
			corners_1v.push_back(corners_1[j]);
			//state[2] = true;
		}
		else {
			cout << "Mis!" << std::endl;
			//state[2] = false;
		}
	}
	//角点个数足够多
	if (corners_1v.size() >= MIN_CORNERS) {

		corners = corners_1v;

		// 提取仿射变化的矩阵
		cv::Mat transform = estimateRigidTransform(corners_0v, corners_1v, false);//partial affine transform
		/*cv::Mat transform = cv::getAffineTransform(corners_1v, corners_0v);*/
		if (transform.total() > 0) {

			// 根据矩阵计算下一帧的人脸矩形框位置
			Contour2f boxCoords;
			boxCoords.push_back(box.tl());
			boxCoords.push_back(box.br());
			Contour2f transformedBoxCoords;
			cv::transform(boxCoords, transformedBoxCoords, transform);
			box = cv::Rect(transformedBoxCoords[0], transformedBoxCoords[1]);
			box.tl() = cv::Point(max(0, box.tl().x), max(0, box.tl().y));

		}

	}
	else {//角点不够的话  重新进行人脸检测  并重置信号队列
		cout << "Tracking failed! Not enough corners left." << endl;
		invalidateFace();
	}
}

//重置信号队列
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
	deque<result>empty;
	swap(empty, hrResult);
	frameCount = 0;
	slideWindowStepCount = 0;
}


//对人脸进行区域分割  分别为 左脸 鼻子 右脸 额头 全脸
void RPPG::makeframedata(const cv::Mat& frameRGB, std::vector<cv::Point>ptr, std::vector<cv::Mat>& processframe)
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

	//左脸的特征点
	cv::Point2i distance = ptr[29] - ptr[27];
	for (int i = 0; i < 8; i++)ptr1.push_back(ptr[i]);
	ptr1.push_back(ptr[48]);
	ptr1.push_back(ptr[31]);
	ptr1.push_back(ptr[27]);
	ptr1.push_back(ptr[21]);
	ptr1.push_back(ptr[20]);
	ptr1.push_back(ptr[19]);
	ptr1.push_back(ptr[18]);


	//鼻子的特征点
	ptr2.push_back(ptr[27]);
	ptr2.push_back(ptr[31]);
	ptr2.push_back(ptr[35]);

	//右脸的特征点
	for (int i = 8; i < 17; i++)ptr3.push_back(ptr[i]);
	for (int i = 26; i > 21; i--)ptr3.push_back(ptr[i]);
	ptr3.push_back(ptr[27]);
	ptr3.push_back(ptr[35]);
	ptr3.push_back(ptr[54]);

	//额头的特征点
	for (int i = 17; i < 27; i++)ptr4.push_back(ptr[i]);
	cv::Point2i noselength = 1.5 * (ptr[27] - ptr[30]);
	ptr4.push_back(ptr[26] + noselength);
	ptr4.push_back(ptr[17] + noselength);

	//嘴唇
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

	//提取左脸
	fillPoly(mask1, vpts1, cv::Scalar(255, 255, 255), 8, 0);
	masks.push_back(mask1);
	//提取鼻子
	fillPoly(mask2, vpts2, cv::Scalar(255, 255, 255), 8, 0);
	masks.push_back(mask2);
	//提取右脸
	fillPoly(mask3, vpts3, cv::Scalar(255, 255, 255), 8, 0);
	masks.push_back(mask3);
	//提取额头
	fillPoly(mask4, vpts4, cv::Scalar(255, 255, 255), 8, 0);
	masks.push_back(mask4);
	//提取全脸
	fillPoly(mask5, vpts5, cv::Scalar(255, 255, 255), 8, 0);
	masks.push_back(mask5);

	//cv::Mat final1 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);
	//cv::Mat final2 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);
	//cv::Mat final3 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);
	//cv::Mat final4 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);
	cv::Mat final5 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);

	//bitwise_and(src, src, final1, mask1);
	//bitwise_and(src, src, final2, mask2);
	//bitwise_and(src, src, final3, mask3);
	//bitwise_and(src, src, final4, mask4);
	bitwise_and(src, src, final5, mask5);


	//processframe.push_back(final1);
	//processframe.push_back(final2);
	//processframe.push_back(final3);
	//processframe.push_back(final4);
	processframe.push_back(final5);

}
void RPPG::histOfArea(cv::Mat face, cv::Mat mask) {
	vector<int> hist(256);
	cv::Mat GRayface, GRaymask;
	cout << "size of mask: " << mask.size() << endl;
	cvtColor(face, GRayface, cv::COLOR_BGR2GRAY);
	//cvtColor(mask, GRaymask, cv::COLOR_BGR2GRAY);
	//cout << "???" << endl;

	for (int i = 0; i < face.rows; ++i) {
		for (int j = 0; j < face.cols; ++j) {
			int maskTemp = (int)mask.at<cv::Vec3b>(i, j)[0];
			if (maskTemp == 255) {
				int faceTemp = (int)face.at<uchar>(i, j);
				hist[faceTemp]++;
			}
		}
	}
	double sum = 0.0, sumNum = 0.0, mean = 0.0, var = 0.0;
	for (int i = 0; i < 256; ++i) {
		if (hist[i] != 0) {
			sum += i * hist[i];
			sumNum += hist[i];
		}
	}
	mean = sum / sumNum;
	for (int i = 0; i < 256; ++i) {
		var += abs(i - mean) * hist[i];
	}
	var /= sumNum;
	cout << "lightness var: " << var << endl;
	drawHist(hist);
}

void RPPG::drawHist(vector<int> nums)
{
	cv::Mat hist = cv::Mat::zeros(600, 800, CV_8UC3);
	auto Max = max_element(nums.begin(), nums.end());//max??????????,??????
	putText(hist, "Histogram", cv::Point(150, 100), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255));
	//*********?????????************//
	cv::Point o = cv::Point(100, 550);
	cv::Point x = cv::Point(700, 550);
	cv::Point y = cv::Point(100, 150);
	//x??
	line(hist, o, x, cv::Scalar(255, 255, 255), 2, 8, 0);
	//y??
	line(hist, o, y, cv::Scalar(255, 255, 255), 2, 8, 0);

	//********??????????***********//
	cv::Point pts[256];
	//?????????
	for (int i = 0; i < 256; i++)
	{
		pts[i].x = i * 2 + 100;
		pts[i].y = 550 - int(nums[i] * (300.0 / (*Max)));//???????[0, 300]
		//?????????
		if ((i + 1) % 16 == 0)
		{
			string num = cv::format("%d", i + 1);
			putText(hist, num, cv::Point(pts[i].x, 570), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
		}
	}
	//??????
	for (int i = 1; i < 256; i++)
	{
		line(hist, pts[i - 1], pts[i], cv::Scalar(0, 255, 0), 2);
	}
	//??????
	imshow("histgram", hist);
}

void RPPG::brightnessException(cv::Mat InputImg, float& cast, float& da)
{
	cv::Mat GRAYimg;
	cvtColor(InputImg, GRAYimg, cv::COLOR_BGR2GRAY);
	float a = 0;
	int Hist[256];
	for (int i = 0; i < 256; i++)
		Hist[i] = 0;
	for (int i = 0; i < GRAYimg.rows; i++)
	{
		for (int j = 0; j < GRAYimg.cols; j++)
		{
			a += float(GRAYimg.at<uchar>(i, j) - 128);//????????????????128?????????
			int x = GRAYimg.at<uchar>(i, j);
			Hist[x]++;
		}
	}
	da = a / float(GRAYimg.rows * InputImg.cols);
	float D = abs(da);
	float Ma = 0;
	for (int i = 0; i < 256; i++)
	{
		Ma += abs(i - 128 - da) * Hist[i];
	}
	Ma /= float((GRAYimg.rows * GRAYimg.cols));
	float M = abs(Ma);
	float K = D / M;
	cast = K;
	return;
}
