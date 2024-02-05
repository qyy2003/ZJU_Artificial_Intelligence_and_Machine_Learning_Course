import torch
import numpy as np
import cv2

from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt  # 展示图片
from torchvision.transforms import transforms

try:
    from MTCNN.detector import FaceDetector
    from MobileNet import MobileNet
except:
    from .MTCNN.detector import FaceDetector
    from .MobileNet import MobileNet

def plot_image(image, image_title="", is_axis=False):
    """
    展示图像
    :param image: 展示的图像，一般是 np.array 类型
    :param image_title: 展示图像的名称
    :param is_axis: 是否需要关闭坐标轴，默认展示坐标轴
    :return:
    """
    # 展示图片
    plt.imshow(image)

    # 关闭坐标轴,默认关闭
    if not is_axis:
        plt.axis('off')

    # 展示受损图片的名称
    plt.title(image_title)

    # 展示图片
    plt.show()


class Recognition(object):
    classes = ["mask", "no_mask"]

    # def __init__(self, mobilenet_path="./results/test.pth"):
    def __init__(self, model_path=None):
        """
        :param: mobilenet_path: XXXX.pth
        """
        self.detector = FaceDetector()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mobilenet = MobileNet(classes=2)
        if model_path:
            self.mobilenet.load_state_dict(
                torch.load(model_path, map_location=device))

    def face_recognize(self, image):
        # 绘制并保存标注图
        drawn_image = self.detector.draw_bboxes(image)
        return drawn_image

    def new_box(self,box):
        print(box)
        x=np.array([box[0],box[1]])
        y=np.array([box[2],box[3]])
        middle=(x+y)/2
        err=(y-x)/2;
        err=err*1.1
        new_x=middle-err
        new_y=middle+err
        print([new_x[0],new_x[1],new_y[0],new_y[1],box[4:]])
        return [new_x[0],new_x[1],new_y[0],new_y[1],box[4:]]
    def mask_recognize(self, image):
        b_boxes, landmarks = self.detector.detect(image)
        detect_face_img = self.detector.draw_bboxes(image)
        face_num = len(b_boxes)
        mask_num = 0
        for box in b_boxes:
            box=self.new_box(box)
            face = image.crop(tuple(box[:4]))
            face=transforms.Resize((160, 160))(face)

            # input_tensor = transforms.ToTensor()(face).to(torch.device('cpu')).numpy()
            # in_arr = np.transpose(input_tensor, (1, 2, 0))  # 将(c,w,h)转换为(w,h,c)。但此时如果是全精度模型，转化出来的dtype=float64 范围是[0,1]。后续要转换为cv2对象，需要乘以255
            # cvimg = cv2.cvtColor(np.uint8(in_arr * 255), cv2.COLOR_RGB2BGR)
            # cv2.imshow("one",cvimg)
            # cv2.waitKey(0)

            # print(box)
            # face = np.array(face)
            face = transforms.ToTensor()(face).unsqueeze(0)
            self.mobilenet.eval()
            with torch.no_grad():
                predict_label = self.mobilenet(face).cpu().data.numpy()
            # print(predict_label)
            current_class = self.classes[np.argmax(predict_label).item()]
            draw = ImageDraw.Draw(detect_face_img)
            if current_class == "mask":
                mask_num += 1
                # font = ImageFont.truetype("consola.ttf", 5, encoding="unic"  )  # 设置字体
                draw.text(((box[0]+box[2])/2, (box[1]+box[3])/2), u'yes', 'fuchsia')
            else:
                # font = ImageFont.truetype("consola.ttf", 5, encoding="unic"  )  # 设置字体
                draw.text(((box[0]+box[2])/2, (box[1]+box[3])/2), u'no', 'fuchsia')

        return detect_face_img, face_num, mask_num


"""
检测人脸，返回人脸位置坐标
其中b_boxes是一个n*5的列表、landmarks是一个n*10的列表，n表示检测出来的人脸个数，数据详细情况如下：
bbox：[左上角x坐标, 左上角y坐标, 右下角x坐标, 右下角y坐标, 检测评分]
landmark：[右眼x, 左眼x, 鼻子x, 右嘴角x, 左嘴角x, 右眼y, 左眼y, 鼻子y, 右嘴角y, 左嘴角y]
"""
if __name__ == "__main__":
    torch.set_num_threads(1)
    detector = FaceDetector()
    img = Image.open("./test1.jpg")
    recognize = Recognition()

    """---detect face--"""
    # draw = recognize.face_recognize(img)
    # plot_image(draw)

    """---crop face ---"""
    draw, all_num, mask_nums = recognize.mask_recognize(img)
    plot_image(draw)
    print("all_num:", all_num, "mask_num", mask_nums)
