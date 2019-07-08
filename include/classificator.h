#pragma once
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class Classificator
{
public:
    vector<string> classesNames;
    virtual Mat Classify(Mat image) = 0 {}
};

class DnnClassificator : public Classificator
{
	string model_path;
	string config_path;
	string labels_path;
	int width;
	int height;
	Scalar mean;
	bool swap;
	Net net;

public:
	DnnClassificator(string m_path, string c_path, string l_path, int im_width, int im_height, Scalar new_mean = (0, 0, 0, 0), bool swapRB = 0);
	Mat Classify(Mat image);
};


