#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "classificator.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ w  width                             |        | image width for classification    }"
"{ h  heigth                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";

int main(int argc, char** argv)
{
	// Process input arguments
	CommandLineParser parser(argc, argv, cmdOptions);
	parser.about(cmdAbout);

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}

	// Load image and init parameters
	
	string path_model(parser.get<String>("model_path"));
	string path_config(parser.get<String>("config_path"));
	string path_label(parser.get<String>("label_path"));
	string path_image(parser.get<String>("i"));
	int width(parser.get<int>("w"));
	int height(parser.get<int>("h"));
	int backendId = DNN_BACKEND_OPENCV;
	int targetID = DNN_TARGET_CPU;
	
	Net net = readNet(path_model, path_config);
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetID);
	
	Mat pic = imread(path_image);
	namedWindow("Deep learning", WINDOW_NORMAL);
	imshow("Deep learning", pic);

	Mat blob;
	double scale = 0.017;
	Size spatial_size = Size(width, height);
	Scalar mean = { 103.94,116.78,123.68 };
	bool swapRB = false;
	bool crop = false;
	int  ddepth = CV_32F;

	blobFromImage(pic, blob, scale, spatial_size, mean, swapRB, crop, ddepth);
	net.setInput(blob);
	Mat prob = net.forward();

	//Image classification
	
		Point classIdPoint;
		double confidence;
		DnnClassificator dst(path_model, path_config, path_label, width, height, mean, swapRB);
		Mat result = dst.Classify(pic);
		minMaxLoc(result, 0, &confidence, 0, &classIdPoint);
		int classId = classIdPoint.x;
	
	//Show result

		minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
		cout << "Class: " << classId << '\n';
		cout << "Confidance " << confidence << '\n';


	return 0;
}
