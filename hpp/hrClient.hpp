
#ifndef hrClient_hpp
#define hrClient_hpp

#include <stdio.h>
#include <iostream>
#include <string>
#include<deque>
#include <algorithm>
#include "RPPG.hpp"

using namespace std;

class hrClient {

public:

	hrClient() {}
	~hrClient() {}

	deque<result> hrResult;
	void isFileExsistence();
	void processonvideo(std::string videoname, int samplingFrequency, int minSignalSize, int maxSignalSize, int chunkSize, bool gui);
	int processoncamera(int cameraindex, int sampletime, int samplingFrequency, int minSignalSize, int maxSignalSize, int chunkSize, bool gui);

	//void processoncamera(int cameraindex, int sampletime, int samplingFrequency, int minSignalSize, int maxSignalSize, int chunkSize, bool gui);
	//void isFileExsistence();




};

#endif /* hrClient */
