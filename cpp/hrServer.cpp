
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
#include <dlib/opencv.h>
#include "Spline.h"
#include "fftw3.h"


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



void hrServer::array2Mat(double green[], int length) {//把客户端传来的绿色通道的数组值变为opencv的MAT格式
	s = cv::Mat1d(length, 1, CV_64F);
	for (int i = 0; i < length; i++) {
		s.at<double>(i, 0) = green[i];
	}
}
void hrServer::array2Mat(int reTrack[], int length) {//把客户端传来的重新人脸跟踪数组变为opencv的MAT格式
	re = cv::Mat1b(length, 1);
	for (int i = 0; i < length; i++) {
		re.at<uchar>(i, 0) = 0;
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



void hrServer::extractSignal_g(int reTrack[], double green[], int length, double fps,int timeStamps[]) {


	double dtimeStamps[1000] = {}, _dtimeStamps[1000], _green[1000];
	for (int i = 0; i < length; ++i) {
		dtimeStamps[i] = (double)timeStamps[i];
	}
	SplineSpace::SplineInterface* sp = new SplineSpace::Spline(dtimeStamps, green, length);
	int lamda = 1;
	sp->AutoInterp(lamda * length, _dtimeStamps, _green);



	array2Mat(reTrack, lamda*length);
	array2Mat(green, lamda*length);
	this->fps = fps*lamda;

	// 差分
	cv::Mat s_den = cv::Mat(s.rows, 1, CV_64F);


	denoise(s, re, s_den);



	// 归一化
	normalization(s_den, s_den);



	// 高通滤波
	cv::Mat s_det = cv::Mat(s_den.rows, s_den.cols, CV_64F);


	detrend(s_den, s_det, this->fps);



	// 低通滤波
	cv::Mat s_mav = cv::Mat(s_det.rows, s_det.cols, CV_64F);

	movingAverage(s_det, s_mav, 3, fmax(floor(this->fps / 6), 2));


	//得到最终处理后的心率信号
	s_mav.copyTo(s_f);
}


//寻找功率谱能量前三的频率
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
double hrServer::estimateHeartrate() {
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
		if (pow(a - b, 2) != 1) {
			if ((((powerSpectrum.at<double>(a, 0) / powerSpectrum.at<double>(b, 0)) >= 1.9) && powerSpectrum.at<double>(a, 0) >= 6 ))
				isbpmgood = 1;
			cout << "峰值的显著性: " << powerSpectrum.at<double>(a, 0) / powerSpectrum.at<double>(b, 0) << endl;

		}
		else
		{
			if ((((powerSpectrum.at<double>(a, 0) / powerSpectrum.at<double>(c, 0)) >= 1.9) && powerSpectrum.at<double>(a, 0) >= 6 ))
				isbpmgood = 1;
			cout << "峰值的显著性: " << powerSpectrum.at<double>(a, 0) / powerSpectrum.at<double>(c, 0) << endl;

		}
		cout << "功率谱能量前三高的频率所在的位置 " << a << " " << powerSpectrum.at<double>(a, 0) << " " << b << " " << powerSpectrum.at<double>(b, 0) << " " << c << " " << powerSpectrum.at<double>(c, 0) << " " << isbpmgood << std::endl;


		double sumEnergy = 0;
		double signalEnergy = pow(powerSpectrum.at<double>(a, 0), 2);
		for (int i = 1; i <= 2; ++i) {
			int leftTemp = a - i < 0 ? 0 : a - i;
			sumEnergy += pow(powerSpectrum.at<double>(leftTemp, 0), 2);
			int rightTemp = a + i > powerSpectrum.rows - 1 ? powerSpectrum.rows - 1 : a + i;
			sumEnergy += pow(powerSpectrum.at<double>(rightTemp, 0), 2);
		}
		snr = 10 * (log(signalEnergy / sumEnergy) / log(10));
		cout << "snr: " << snr << endl;


		minMaxLoc(powerSpectrum, &min, &max, &pmin, &pmax, bandMask); p[0] = pmax.y;


		// 计算出此次心率信号得到的瞬时心率  BPM
		bpm = pmax.y * fps / total * SEC_PER_MIN;
		prominentFrequency = pmax.y * fps / total;
		unitFrequency = fps / total;
		extractHeartrateVariability();

		//如果足够的鲁棒可靠  放入时间队列

		if (isbpmgood) {
			bpmsc.push_back(bpm);
			snrs.push_back(snr);
			hrvs.push_back(hrv);
			lastbpm = bpm;
			lasthrv = hrv;
			lastsnr = snr;
		}
		else if (lastbpm) {
			bpmsc.push_back(lastbpm);
			hrvs.push_back(lasthrv);
			snrs.push_back(lastsnr);
		}


		cout << "all area   " << "FPS=" << fps << " Vals=" << powerSpectrum.rows << " Peak=" << pmax.y << endl;
		if (bpmsc.rows > 0) {
			meanBpm = mean(bpmsc)(0);
			meanHrv = mean(hrvs)(0);
			meanSnr = mean(snrs)(0);
			if (bpmsc.rows > 8) {
				bpmsc.pop_back(bpmsc.rows);
				hrvs.pop_back(hrvs.rows);
			}

		}
		cout << "bpm: " << meanBpm << "瞬时心率： " << bpm << endl;//此处meanBpm不为0可以返回客户端
		cout << "信噪比： " << meanSnr << "db " << "心率变异性： " << meanHrv << "ms" << endl;
		return meanBpm;

	}
}
void hrServer::findPeaks(double* src, int src_lenth, double distance, int* indMax, int* indMax_len)
{
	int* sign = new int[src_lenth];
	int max_index = 0,
		min_index = 0;
	*indMax_len = 0;

	for (int i = 1; i < src_lenth; i++)
	{
		double diff = src[i] - src[i - 1];
		if (diff > 0)          sign[i - 1] = 1;
		else if (diff < 0) sign[i - 1] = -1;
		else                sign[i - 1] = 0;
	}
	for (int j = 1; j < src_lenth - 1; j++)
	{
		double diff = sign[j] - sign[j - 1];
		if (diff < 0)      indMax[max_index++] = j;
	}

	int* flag_max_index = new int[max_index];
	int* idelete = new int[max_index];
	int* temp_max_index = new int[max_index];
	int bigger = 0;
	double tempvalue = 0;
	int i, j, k;
	//波峰  
	for (int i = 0; i < max_index; i++)
	{
		flag_max_index[i] = 0;
		idelete[i] = 0;
	}
	for (i = 0; i < max_index; i++)
	{
		tempvalue = -1;
		for (j = 0; j < max_index; j++)
		{
			if (!flag_max_index[j])
			{
				if (src[indMax[j]] > tempvalue)
				{
					bigger = j;
					tempvalue = src[indMax[j]];
				}
			}
		}
		flag_max_index[bigger] = 1;
		if (!idelete[bigger])
		{
			for (k = 0; k < max_index; k++)
			{
				idelete[k] |= (indMax[k] - distance <= indMax[bigger] & indMax[bigger] <= indMax[k] + distance);
			}
			idelete[bigger] = 0;
		}
	}
	for (i = 0, j = 0; i < max_index; i++)
	{
		if (!idelete[i])
			temp_max_index[j++] = indMax[i];
	}
	for (i = 0; i < max_index; i++)
	{
		if (i < j)
			indMax[i] = temp_max_index[i];
		else
			indMax[i] = 0;
	}
	max_index = j;


	*indMax_len = max_index;

	delete sign;
	delete flag_max_index;
	delete temp_max_index;
	delete idelete;
}
double* hrServer::ideal_bandpass_filter(double* input, int length, float fl, float fh, float fps) {
	fftw_plan p, fp;
	fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);
	fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);
	fftw_complex* fft_w = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);
	double* output = (double*)fftw_malloc(sizeof(double) * length);
	for (int i = 0; i < length; i++) {
		in[i][0] = input[i];
		in[i][1] = 0.0;
	}
	//FFT
	p = fftw_plan_dft_1d(length, in, fft_w, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p); /* repeat as needed*/
	int mid = length / 2;
	//IDEAL BANDPASS
	double delta = fps / length;
	int fl_local = int(fl / delta);
	int fh_local = int(fh / delta);
	for (int i = 0; i < fl_local; i++) {
		fft_w[i][0] = 0; fft_w[i][1] = 0;
		fft_w[length - 1 - i][0] = 0; fft_w[length - 1 - i][1] = 0;
	}
	for (int i = fh_local; i <= mid; i++) {
		fft_w[i][0] = 0; fft_w[i][1] = 0;
		fft_w[length - 1 - i][0] = 0; fft_w[length - 1 - i][1] = 0;
	}
	for (int i = 0; i < length; i++) {
		fft_w[i][0] /= length;
		fft_w[i][1] /= length;
	}
	//DTFT
	fp = fftw_plan_dft_1d(length, fft_w, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(fp); /* repeat as needed*/
	for (int j = 0; j < length; j++) {
		output[j] = out[j][0];
	}
	fftw_destroy_plan(p);
	fftw_destroy_plan(fp);
	fftw_free(in);
	fftw_free(out);
	fftw_free(fft_w);
	return output;
}

void hrServer::extractHeartrateVariability() {

	//心率变异性代表的是心跳逐次周期差异的变化情况 这边用sdnn来代替
	//将s_f从mat格式数据转化为double数组
	double s_fdouble[1000] = {};
	int s_fdoubleLen = s_f.rows;
	for (int i = 0; i < s_f.rows; ++i) {
		s_fdouble[i] = s_f.at<double>(i, 0);
		//cout << s_fdouble[i] << endl;
	}


	//对数组进行 带通滤波
	float fl = (prominentFrequency - 1.5 * unitFrequency) > 0 ? prominentFrequency - 1.5 * unitFrequency : 0;
	float fh = prominentFrequency + 1.5 * unitFrequency;
	cout << "fl: " << fl << "fh: " << fh << endl;
	double* _s_fdouble = ideal_bandpass_filter(s_fdouble, s_fdoubleLen, fl, fh, fps);
	//ofstream out1("bandpass.txt");
	//for (int i = 0; i < s_fdoubleLen; ++i) {
	//	out1 << _s_fdouble[i] << "\n";
	//}
	//out1.close();
	//cout << "fps: " << fps << endl;


	//进行峰值检测
	int peaksLen = 100;
	int* peaks = new int[peaksLen];
	int distance = fps / prominentFrequency / 2;
	findPeaks(_s_fdouble, s_fdoubleLen, distance, peaks, &peaksLen);
	double unitTime = 1000 / fps;


	vector<int> peaksDiff;
	for (int i = 1; i < peaksLen; ++i) {
		int tempDiff = peaks[i] - peaks[i - 1];
		peaksDiff.push_back(tempDiff * unitTime);
	}
	//cout << "峰值时间点" << endl;
	double meadDiff = 0;
	int sumDiff = 0;
	for (auto it : peaksDiff) {
		//cout << it << " ";
		sumDiff += it;
	}
	meadDiff = 1.0 * sumDiff / peaksDiff.size();

	//
	double sumDviation = 0.0;
	for (auto it : peaksDiff) {
		sumDviation += pow((it - meadDiff), 2);
	}
	double hrvPeaksNum = 0;
	double hrv = 0;
	cout << "unittime: " << unitTime << endl;
	cout << "sumDviation :" << sumDviation << endl;
	hrvPeaksNum = sqrt(sumDviation / peaksDiff.size());
	//cout << "hrvPeaksNum: " << hrvPeaksNum << endl;
	cout << "hrv: " << hrvPeaksNum << endl;
	this->hrv = hrvPeaksNum;
	delete peaks;
}
