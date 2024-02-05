import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


class FaceDet():
    def __init__(self):
        self.opencv_dnn_path = 'datasets/5f680a696ec9b83bb0037081-momodel/data/mindspore_model_data/opencv_dnn/'
        self.threshold = 0.15
        self.caffe_model = self.opencv_dnn_path + "deploy.prototxt"
        self.caffe_param = self.opencv_dnn_path + "res10_300x300_ssd_iter_140000_fp16.caffemodel"

    def draw_detections(self, image, detections):
        h, w, c = image.shape
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 255, 0), 1)
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        return image

    def detect(self, image):
        net = cv2.dnn.readNetFromCaffe(self.caffe_model, self.caffe_param)

        # def blobFromImage(image, scalefactor=None, size=None, mean=None, swapRB=None, crop=None, ddepth=None)
        # image：输入图像
        # mean：对每个通道像素值减去对应的均值，这里用(104.0, 177.0, 123.0)，和模型训练时的值一致
        # scalefactor：对像素值的缩放比例
        # size：模型输入图片的尺寸
        # swapRB：OpenCV默认的图片通道顺序是BGR，如果需要交换R和G，则设为True
        # crop: 调整图片大小后，是否裁剪
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blob)
        detections = net.forward()
        return detections


if __name__=='__main__':
    img = cv2.imread("test.jpg")
    detect = FaceDet()
    detections = detect.detect(img)
    drawed_img = detect.draw_detections(img, detections)

    # OpenCV reads image to BGR format. Transform images before showing it.
    drawed_img = cv2.cvtColor(drawed_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(drawed_img)
    plt.show()