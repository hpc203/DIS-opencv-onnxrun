#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;


class DIS
{
public:
	DIS(string model_path);
	Mat detect(Mat srcimg);
private:
	vector<float> input_image_;
	int inpWidth;
	int inpHeight;
	int outWidth;
	int outHeight;
	const float score_th = 0;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "DIS");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

DIS::DIS(string model_path)
{
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->outHeight = output_node_dims[0][2];
	this->outWidth = output_node_dims[0][3];
}

Mat DIS::detect(Mat srcimg)
{
	Mat dstimg;
	resize(srcimg, dstimg, Size(this->inpWidth, this->inpHeight));
	this->input_image_.resize(this->inpWidth * this->inpHeight * dstimg.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < this->inpHeight; i++)
		{
			for (int j = 0; j < this->inpWidth; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * this->inpHeight * this->inpWidth + i * this->inpWidth + j] = pix / 255.0 - 0.5;
			}
		}
	}
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	float* pred = ort_outputs[0].GetTensorMutableData<float>();
	Mat mask(outHeight, outWidth, CV_32FC1, pred);
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