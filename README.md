# DIS-opencv-onnxrun
分别使用OpenCV、ONNXRuntime部署DIS高精度图像二类分割，包含C++和Python两种版本的程序。
本套程序对应的paper是ECCV2022的一篇文章《Highly Accurate Dichotomous Image Segmentation》

本套程序提供了50个onnx文件，占用磁盘空间8.2G，onnx文件在百度云盘，下载链接：https://pan.baidu.com/s/19jENx2Ul8oJn-iLBFK8sLg 
提取码：uphj

需要注意的是opencv不能加载['isnet_general_use_HxW.onnx', 'isnet_HxW.onnx', 'isnet_Nx3xHxW.onnx']这3个文件做推理，
onnxruntime不能加载['isnet_general_use_HxW.onnx', 'isnet_HxW.onnx', 'isnet_Nx3xHxW.onnx']这3个文件做推理
