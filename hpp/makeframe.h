#pragma once
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
void makeframedata(cv::Mat frameRGB, std::vector<cv::Point>ptr, std::vector<cv::Mat>& processframe);
void makeframedata(cv::Mat frameRGB, std::vector<cv::Point>ptr, std::vector<cv::Mat>& processframe)
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

	//left cheek
	cv::Point2i distance = ptr[29] - ptr[27];
	for (int i = 0; i < 9; i++)ptr1.push_back(ptr[i]);
	ptr1.push_back(ptr[48]);
	ptr1.push_back(ptr[31]);
	ptr1.push_back(ptr[27]);
	ptr1.push_back(ptr[21]);
	ptr1.push_back(ptr[20]);
	ptr1.push_back(ptr[19]);
	ptr1.push_back(ptr[18]);
	

	//nose
	ptr2.push_back(ptr[27]);
	ptr2.push_back(ptr[31]);
	ptr2.push_back(ptr[35]);

	//right cheek
	for (int i = 8; i < 17; i++)ptr3.push_back(ptr[i]);
	for (int i = 26; i > 21; i--)ptr3.push_back(ptr[i]);
	ptr3.push_back(ptr[27]);
	ptr3.push_back(ptr[35]);
	ptr3.push_back(ptr[54]);

	//forehead
	for (int i = 17; i < 27; i++)ptr4.push_back(ptr[i]);
	cv::Point2i noselength = 2.5*(ptr[27] - ptr[30]);
	ptr4.push_back(ptr[26]+ noselength);
	ptr4.push_back(ptr[17] + noselength);

	//lip
	for (int i = 48; i < 60; i++)ptr5.push_back(ptr[i]);


	std::vector<std::vector<cv::Point> > vpts1;
	std::vector<std::vector<cv::Point> > vpts2;
	std::vector<std::vector<cv::Point> > vpts3;
	std::vector<std::vector<cv::Point> > vpts4;
	std::vector<std::vector<cv::Point> > vpts5;

	vpts1.push_back(ptr1);
	vpts1.push_back(ptr3);
	vpts1.push_back(ptr2);
	vpts1.push_back(ptr4);
	//left cheek+right cheek
	vpts2.push_back(ptr2);//nose
	
	vpts3.push_back(ptr3);
	                      //lip
	vpts4.push_back(ptr4);//forehead
	vpts5.push_back(ptr1);
	vpts5.push_back(ptr2);
	vpts5.push_back(ptr3);
	vpts5.push_back(ptr4);

	fillPoly(mask1, vpts1, cv::Scalar(255, 255, 255), 8, 0);
	masks.push_back(mask1);
	fillPoly(mask2, vpts2, cv::Scalar(255, 255, 255), 8, 0);
	masks.push_back(mask2);
	fillPoly(mask3, vpts3, cv::Scalar(255, 255, 255), 8, 0);
	masks.push_back(mask3);
	fillPoly(mask4, vpts4, cv::Scalar(255, 255, 255), 8, 0);
	masks.push_back(mask4);

	fillPoly(mask5, vpts5, cv::Scalar(255, 255, 255), 8, 0);
	masks.push_back(mask5);

	cv::Mat final1 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);
	cv::Mat final2 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);
	cv::Mat final3 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);
	cv::Mat final4 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);
	cv::Mat final5 = cv::Mat::zeros(frameRGB.size(), CV_8UC3);

	bitwise_and(src, src, final1, mask1);
	bitwise_and(src, src, final2, mask2);
	bitwise_and(src, src, final3, mask3);
	bitwise_and(src, src, final4, mask4);
	bitwise_and(src, src, final5, mask5);


	processframe.push_back(final1);
	processframe.push_back(final2);
	processframe.push_back(final3);
	processframe.push_back(final4);
	processframe.push_back(final5);

	cv::imshow("area1", final1);
	cv::imshow("area2", final2);
	cv::imshow("area3", final3);
	cv::imshow("area4", final4);
	cv::imshow("area5", final5);
	
}