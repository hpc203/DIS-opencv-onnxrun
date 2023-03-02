#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace dnn;


class DIS
{
public:
	DIS(string model_path);
	Mat detect(Mat srcimg);
private:
	int inpWidth;
	int inpHeight;
	const float score_th = 0;

	Net net;
};

DIS::DIS(string model_path)
{
	this->net = readNet(model_path);

	size_t pos = model_path.rfind("_");
	size_t pos_ = model_path.rfind(".");
	int len = pos_ - pos - 1;
	string hxw = model_path.substr(pos + 1, len);

	pos = hxw.rfind("x");
	string h = hxw.substr(0, pos);
	len = hxw.length() - pos;
	string w = hxw.substr(pos + 1, len);
	this->inpHeight = stoi(h);
	this->inpWidth = stoi(w);
}

Mat DIS::detect(Mat srcimg)
{
	Mat blob = blobFromImage(srcimg, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0.5, 0.5, 0.5), true, false);
	/*Mat img;
	resize(srcimg, img, Size(this->inpWidth, this->inpHeight), INTER_LINEAR);
	cvtColor(img, img, COLOR_BGR2RGB);
	img.convertTo(img, CV_32FC3, 1.0 / 255.0, -0.5);
	Mat blob = blobFromImage(img);*/

	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());   // 开始推理

	Mat mask(outs[0].size[2], outs[0].size[3], CV_32FC1, (float*)outs[0].data);
	double min_value, max_value;
	minMaxLoc(mask, &min_value, &max_value, 0, 0);
	mask = (mask - min_value) / (max_value - min_value);
	///compare(mask, score_th, mask, cv::CMP_GT);   /////比较结果为真的地方值为 255，否则为0；
	mask *= 255;
	mask.convertTo(mask, CV_8UC1);
	resize(mask, mask, Size(srcimg.cols, srcimg.rows));

	return mask;
}

Mat generate_overlay_image(Mat srcimg, Mat mask)
{
	Mat overlay_image = srcimg.clone();
	for (int i = 0; i < srcimg.rows; i++)
	{
		for (int j = 0; j < srcimg.cols; j++)
		{
			if (mask.ptr<uchar>(i)[j] == 0)
			{
				overlay_image.at<Vec3b>(i, j)[0] = 255;
				overlay_image.at<Vec3b>(i, j)[1] = 255;
				overlay_image.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	return overlay_image;
}

int main()
{
	DIS mynet("weights/isnet_general_use_480x640.onnx");
	string imgpath = "images/bike.jpg";
	Mat srcimg = imread(imgpath);
	
	Mat mask = mynet.detect(srcimg);
	Mat overlay_image = generate_overlay_image(srcimg, mask);
	

	namedWindow("srcimg", WINDOW_NORMAL);
	imshow("srcimg", srcimg);
	namedWindow("mask", WINDOW_NORMAL);
	imshow("mask", mask);
	namedWindow("overlay_image", WINDOW_NORMAL);
	imshow("overlay_image", overlay_image);
	waitKey(0);
	destroyAllWindows();
}