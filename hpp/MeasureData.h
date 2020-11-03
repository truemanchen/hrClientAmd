
#pragma once
//#include <array>
//#include <deque>
#include <opencv2/opencv.hpp>
typedef int64 TimerTimestamp;

template <typename VAL>
class Measure
{
public:
	TimerTimestamp t; // ����ʱ��
	VAL val;		  // ʵ��ֵ

	Measure(TimerTimestamp t_, const VAL& val_)
	{
		t = t_;
		val = val_;
	}
};