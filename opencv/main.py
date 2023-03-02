import argparse
import cv2
import numpy as np
###opencv dnn load ['isnet_general_use_HxW.onnx', 'isnet_HxW.onnx', 'isnet_Nx3xHxW.onnx']  inference failed
class DIS():
    def __init__(self, modelpath, score_th=None):
        self.net = cv2.dnn.readNet(modelpath)
        hxw = modelpath.split('_')[-1].split('.')[0].split('x')
        self.input_height = int(hxw[0])
        self.input_width = int(hxw[1])
        self.score_th = score_th
        self.output_names = self.net.getUnconnectedOutLayersNames()

    def detect(self, srcimg):
        img = cv2.resize(srcimg, dsize=(self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0 - 0.5
        blob = cv2.dnn.blobFromImage(img)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_names)
        
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
    parser.add_argument("--imgpath", type=str, default='images/bike.jpg')
    parser.add_argument("--modelpath", type=str, default='weights/isnet_general_use_480x640.onnx')
    args = parser.parse_args()
    
    mynet = DIS(args.modelpath)
    srcimg = cv2.imread(args.imgpath)
    mask = mynet.detect(srcimg)
    mask, overlay_image = generate_overlay_image(srcimg, mask)

    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, np.hstack((srcimg, mask)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()