import argparse
import cv2
import numpy as np
import onnxruntime
### onnxruntime load ['isnet_general_use_HxW.onnx', 'isnet_HxW.onnx', 'isnet_Nx3xHxW.onnx']  inference failed
class DIS():
    def __init__(self, modelpath, score_th=None):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.net = onnxruntime.InferenceSession(modelpath, so)
        self.input_height = self.net.get_inputs()[0].shape[2]
        self.input_width = self.net.get_inputs()[0].shape[3]
        self.input_name = self.net.get_inputs()[0].name
        self.output_name = self.net.get_outputs()[0].name
        self.score_th = score_th

    def detect(self, srcimg):
        img = cv2.resize(srcimg, dsize=(self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0 - 0.5
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        outs = self.net.run([self.output_name], {self.input_name: blob})
        
        mask = np.array(outs[0]).squeeze()
        min_value = np.min(mask)
        max_value = np.max(mask)
        mask = (mask - min_value) / (max_value - min_value)
        if self.score_th is not None:
            mask = np.where(mask < self.score_th, 0, 1)
        mask *= 255
        mask = mask.astype('uint8')

        mask = cv2.resize(mask, dsize=(srcimg.shape[1], srcimg.shape[0]), interpolation=cv2.INTER_LINEAR)
        return mask

def generate_overlay_image(srcimg, mask):
    overlay_image = np.zeros(srcimg.shape, dtype=np.uint8)
    overlay_image[:] = (255, 255, 255)
    mask = np.stack((mask,) * 3, axis=-1).astype('uint8')  ###沿着通道方向复制3次
    mask_image = np.where(mask, srcimg, overlay_image)
    return mask, mask_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='images/cam_image47.jpg')
    parser.add_argument("--modelpath", type=str, default='weights/isnet_general_use_480x640.onnx')
    args = parser.parse_args()
    
    mynet = DIS(args.modelpath)
    srcimg = cv2.imread(args.imgpath)
    mask = mynet.detect(srcimg)
    mask, overlay_image = generate_overlay_image(srcimg, mask)

    winName = 'Deep learning object detection in onnxruntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, np.hstack((srcimg, mask)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()