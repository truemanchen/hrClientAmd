
#include "hrServer.hpp"
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
//dlib::shape_predictor sp;
//dlib::full_object_detection shape, shape1;
//string modelpath = "shape_predictor_68_face_landmarks.dat";



void hrServer::array2Mat(double green[], int length) {//把客户端传来的绿色通道的数组值变为opencv的MAT格式
	s = cv::Mat1d(length, 1, CV_64F);
	for (int i = 0; i < length; i++) {
		s.at<double>(i, 0) = green[i];
	}
}
void hrServer::array2Mat(int reTrack[], int length) {//把客户端传来的重新人脸跟踪数组变为opencv的MAT格式
	re = cv::Mat1b(length, 1);
	for (int i = 0; i < length; i++) {
		re.at<uchar>(i, 0) = reTrack[i];
	}

}



/* 滤波 */

// 把心率信号进行归一化    ➖平均值 ➗方差
void hrServer::normalization(cv::InputArray _a, cv::OutputArray _b) {
	_a.getMat().copyTo(_b);
	cv::Mat b = _b.getMat();
	cv::Scalar mean, stdDev;
	for (int i = 0; i < b.cols; i++) {
		meanStdDev(b.col(i), mean, stdDev);
		b.col(i) = (b.col(i) - mean[0]) / stdDev[0];
	}
}

// 进行差分 放大变化 减小每帧的不均匀
void hrServer::denoise(cv::InputArray _a, cv::InputArray _jumps, cv::OutputArray _b) {

	cv::Mat a = _a.getMat().clone();
	cv::Mat jumps = _jumps.getMat().clone();

	CV_Assert(a.type() == CV_64F && jumps.type() == CV_8U);

	if (jumps.rows != a.rows) {
		jumps.rowRange(jumps.rows - a.rows, jumps.rows).copyTo(jumps);
	}

	cv::Mat diff;
	subtract(a.rowRange(1, a.rows), a.rowRange(0, a.rows - 1), diff);
	for (int i = 1; i < jumps.rows; i++) {
		if (jumps.at<bool>(i, 0)) {
			cv::Mat mask = cv::Mat::zeros(a.size(), CV_8U);
			mask.rowRange(i, mask.rows).setTo(cv::Scalar(1));
			for (int j = 0; j < a.cols; j++) {
				add(a.col(j), -diff.at<double>(i - 1, j), a.col(j), mask.col(j));
				///*for (int i = 0;i < a.rows;i++) {
				//	std::cout <<"a "<< i <<" "<< a.at<double>(i, 0) << endl;*/

				//}
			}
		}
	}

	a.copyTo(_b);
}

// 消除心率信号的基线漂移  并高通滤波
void hrServer::detrend(cv::InputArray _a, cv::OutputArray _b, int lambda) {

	cv::Mat a = _a.getMat();
	CV_Assert(a.type() == CV_64F);

	// Number of rows
	int rows = a.rows;

	if (rows < 3) {
		a.copyTo(_b);
	}
	else {
		// 构造 I
		cv::Mat i = cv::Mat::eye(rows, rows, a.type());
		// 构造 D2
		cv::Mat d = cv::Mat(cv::Matx<double, 1, 3>(1, -2, 1));
		cv::Mat d2Aux = cv::Mat::ones(rows - 2, 1, a.type()) * d;
		cv::Mat d2 = cv::Mat::zeros(rows - 2, rows, a.type());
		for (int k = 0; k < 3; k++) {
			d2Aux.col(k).copyTo(d2.diag(k));
		}
		// 心率信号矩阵 b = (I - (I + λ^2 * D2^t*D2)^-1) * a
		cv::Mat b = (i - (i + double(lambda) * double(lambda) * d2.t() * d2).inv()) * a;
		b.copyTo(_b);
	}
}

// 均值滤波
void hrServer::movingAverage(cv::InputArray _a, cv::OutputArray _b, int n, int s) {

	CV_Assert(s > 0);

	_a.getMat().copyTo(_b);
	cv::Mat b = _b.getMat();
	for (size_t i = 0; i < n; i++) {
		cv::blur(b, b, cv::Size(s, s));
	}
}
//时域转换成频域
void hrServer::timeToFrequency(cv::InputArray _a, cv::OutputArray _b, bool magnitude) {

	// Prepare planes
	cv::Mat a = _a.getMat();
	cv::Mat planes[] = { cv::Mat_<float>(a), cv::Mat::zeros(a.size(), CV_32F) };
	cv::Mat powerSpectrum;
	merge(planes, 2, powerSpectrum);

	// 傅里叶变换
	dft(powerSpectrum, powerSpectrum, cv::DFT_COMPLEX_OUTPUT);

	if (magnitude) {
		split(powerSpectrum, planes);
		cv::magnitude(planes[0], planes[1], planes[0]);
		planes[0].copyTo(_b);
	}
	else {
		powerSpectrum.copyTo(_b);
	}
}



void hrServer::extractSignal_g(int reTrack[], double green[], int length, double fps) {
	array2Mat(reTrack, length);
	array2Mat(green, length);
	this->fps = fps;

	// 差分
	cv::Mat s_den = cv::Mat(s.rows, 1, CV_64F);
	//cv::Mat s_den1 = cv::Mat(s.rows, 1, CV_64F);
	//cv::Mat s_den2 = cv::Mat(s.rows, 1, CV_64F);
	//cv::Mat s_den3 = cv::Mat(s.rows, 1, CV_64F);
	//cv::Mat s_den4 = cv::Mat(s.rows, 1, CV_64F);
	denoise(s, re, s_den);
	//denoise(s1.col(1), re, s_den1);
	//denoise(s2.col(1), re, s_den2);
	//denoise(s3.col(1), re, s_den3);
	//denoise(s4.col(1), re, s_den4);

	// 归一化
	normalization(s_den, s_den);
	//normalization(s_den1, s_den1);
	//normalization(s_den2, s_den2);
 //   normalization(s_den3, s_den3);
	//normalization(s_den4, s_den4);

	// 高通滤波
	cv::Mat s_det = cv::Mat(s_den.rows, s_den.cols, CV_64F);
	//cv::Mat s_det1 = cv::Mat(s_den1.rows, s_den1.cols, CV_64F);
	//cv::Mat s_det2 = cv::Mat(s_den2.rows, s_den2.cols, CV_64F);
	//cv::Mat s_det3 = cv::Mat(s_den3.rows, s_den3.cols, CV_64F);
	//cv::Mat s_det4 = cv::Mat(s_den4.rows, s_den4.cols, CV_64F);
	//int time4 = (cv::getTickCount() * 1000.0) / cv::getTickFrequency();
	//cout << "time4:" << time4 << std::endl;
	detrend(s_den, s_det, fps);
	//int time5 = (cv::getTickCount() * 1000.0) / cv::getTickFrequency();
	//cout << "time5:" << time5 << std::endl;
	//detrend(s_den1, s_det1, fps);
	//detrend(s_den2, s_det2, fps);
	//detrend(s_den3, s_det3, fps);
	//detrend(s_den4, s_det4, fps);

	// 低通滤波
	cv::Mat s_mav = cv::Mat(s_det.rows, s_det.cols, CV_64F);
	//cv::Mat s_mav1 = cv::Mat(s_det.rows, s_det.cols, CV_64F);
	//cv::Mat s_mav2 = cv::Mat(s_det.rows, s_det.cols, CV_64F);
	//cv::Mat s_mav3 = cv::Mat(s_det.rows, s_det.cols, CV_64F);
	//cv::Mat s_mav4 = cv::Mat(s_det.rows, s_det.cols, CV_64F);
	movingAverage(s_det, s_mav, 3, fmax(floor(fps / 6), 2));
	//movingAverage(s_det1, s_mav1, 3, fmax(floor(fps / 6), 2));
	//movingAverage(s_det2, s_mav2, 3, fmax(floor(fps / 6), 2));
	//movingAverage(s_det3, s_mav3, 3, fmax(floor(fps / 6), 2));
	//movingAverage(s_det4, s_mav4, 3, fmax(floor(fps / 6), 2));


	//得到最终处理后的心率信号
	s_mav.copyTo(s_f);
	//s_mav1.copyTo(s_f1);
	//s_mav2.copyTo(s_f2);
	//s_mav3.copyTo(s_f3);
	//s_mav4.copyTo(s_f4);

	// Logging
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
void hrServer::findTriMax(cv::Mat powerSpetrum) {
	cv::Mat powerSpectrum_copy = powerSpetrum.clone();
	double max;
	memset(inds, 0, 3);
	int start = min(low, s_f.rows);
	int end = min(high, s_f.rows) + 1;
	for (int i = 0; i < 3; i++) {
		max = 0.0; double v; int p;
		for (int j = start; j < end; j++) {
			v = powerSpectrum_copy.at<double>(j, 0);
			if (max < v) {
				max = v;
				p = j;
			}
		}
		inds[i] = p;
		powerSpectrum_copy.at<double>(p, 0) = 0.0;
	}

}



//把处理后的心率信号 进行功率谱变化 求取能量最高的频率  即为心率
void hrServer::estimateHeartrate() {
	isbpmgood = 0;
	powerSpectrum = cv::Mat(s_f.size(), CV_32F);


	timeToFrequency(s_f, powerSpectrum, true);

	low = (int)(s.rows * LOW_BPM / SEC_PER_MIN / fps);
	high = (int)(s.rows * HIGH_BPM / SEC_PER_MIN / fps) + 1;



	// 正常人的心率带通
	const int total = s_f.rows;
	cv::Mat bandMask = cv::Mat::zeros(s_f.size(), CV_8U);
	int start = min(low, total);
	int end = min(high, total) + 1;
	bandMask.rowRange(start, end).setTo(cv::Scalar(1));

	if (!powerSpectrum.empty()) {

		// 功率谱的最低和最高值
		double min, max;


		cv::Point pmin, pmax;


		int p[5], maxp = 0;
		double distance[5], maxdistance = 0.0;
		findTriMax(powerSpectrum);
		int  a = inds[0], b = inds[1], c = inds[2];
		cout << "峰值的显著性: " << powerSpectrum.at<double>(a, 0) / powerSpectrum.at<double>(b, 0) << endl;
		if (pow(a - b, 2) != 1) {
			if (((powerSpectrum.at<double>(a, 0) / powerSpectrum.at<double>(b, 0) >= 1.9) && powerSpectrum.at<double>(a, 0) >= 6 && powerSpectrum.at<double>(b, 0) >= 6))
				isbpmgood = 1;
		}
		else
		{
			if (((powerSpectrum.at<double>(a, 0) / powerSpectrum.at<double>(c, 0) >= 1.9) && powerSpectrum.at<double>(a, 0) >= 6 && powerSpectrum.at<double>(c, 0) >= 6))
				isbpmgood = 1;
		}
		cout << "功率谱能量前三高的频率所在的位置 " << a << " " << powerSpectrum.at<double>(a, 0) << " " << b << " " << powerSpectrum.at<double>(b, 0) << " " << c << " " << powerSpectrum.at<double>(c, 0) << " " << isbpmgood << std::endl;


		//for (int i = 0; i < 5; i++) {
		//	if (maxdistance < distance[i]) {
		//		maxdistance = distance[i];
		//		maxp = i;
		//	}
		//}

		minMaxLoc(powerSpectrum, &min, &max, &pmin, &pmax, bandMask); p[0] = pmax.y;


		// 计算出此次心率信号得到的瞬时心率  BPM
		bpm = pmax.y * fps / total * SEC_PER_MIN;


		//如果足够的鲁棒可靠  放入时间队列

		if (isbpmgood) {
			bpmsc.push_back(bpm);
			lastbpm = bpm;
		}
		else if (lastbpm) {
			bpmsc.push_back(lastbpm);
		}


		cout << "all area   " << "FPS=" << fps << " Vals=" << powerSpectrum.rows << " Peak=" << pmax.y << endl;
		if (bpmsc.rows > 8) {
			meanBpm = mean(bpmsc)(0);
			bpmsc.pop_back(2 * bpmsc.rows / 3);
		}
		cout << "bpm: " << meanBpm << endl;//此处meanBpm不为0可以返回客户端


	}
}

