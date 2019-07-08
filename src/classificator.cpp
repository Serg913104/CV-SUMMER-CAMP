#include "classificator.h"
DnnClassificator::DnnClassificator(string m_path, string c_path, string l_path, int im_width, int im_height, Scalar new_mean, bool swapRB)
{
	string model_path = m_path;
	string config_path = c_path;
	string labels_path = l_path;
	int width = im_width;
	int height = im_height;
	Scalar mean = new_mean;
	bool swap = swapRB;

	net = readNet(model_path, config_path);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
}

Mat DnnClassificator::Classify(Mat image)
{
	Mat tensor, result;
	blobFromImage(image, tensor, 1, Size(224, 224), mean, swap, false);
	net.setInput(tensor);
	result = net.forward();
	result = result.reshape(1, 1);
	return result;
}