import warnings
# 忽视警告
warnings.filterwarnings('ignore')

import cv2
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from MobileNet import MobileNet

# from torch_py.Utils import plot_image
# from torch_py.MTCNN.detector import FaceDetector
# from torch_py.MobileNet import MobileNet
# from torch_py.FaceRec import Recognition

# 垃圾分类数据集标签，以及用于标签映射的字典。
index = {'00_00': 0, '00_01': 1, '00_02': 2, '00_03': 3, '00_04': 4, '00_05': 5, '00_06': 6, '00_07': 7,
         '00_08': 8, '00_09': 9, '01_00': 10, '01_01': 11, '01_02': 12, '01_03': 13, '01_04': 14,
         '01_05': 15, '01_06': 16, '01_07': 17, '02_00': 18, '02_01': 19, '02_02': 20, '02_03': 21,
         '03_00': 22, '03_01': 23, '03_02': 24, '03_03': 25}
inverted = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle',
            6: 'Cardboard', 7: 'Basketball',
            8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror',
            14: 'Toothbrush',
            15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery',
            20: 'Fluorescent lamp', 21: 'Tablet capsules',
            22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}
# def my_transforms():
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device="cpu"


model = MobileNet(classes=26).to(device)
model.load_state_dict(
                torch.load('./results/temp2.pth', map_location=device))
model.eval()
def processing_data(img, height=224, width=224):
    """
    数据处理部分
    :param data_path: 数据路径
    :param height:高度
    :param width: 宽度
    :param batch_size: 每次读取图片的数量
    :param test_split: 测试集划分比例
    :return:
    """
    transforms = T.Compose([
        T.Resize((height, width)),
        # T.RandomRotation(15,center=(height/2,width/2)),
        # T.RandomResizedCrop((height, width),scale = (0.7,1.0)),
        # T.Random
        # T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        # T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        T.ToTensor(),  # 转化为张量
        T.Normalize([0], [1]),  # 归一化
    ])
    plt.imshow(img)
    plt.show()
    return transforms(img)

def predict(img):
    if isinstance(img, np.ndarray):
        # 转化为 PIL.JpegImagePlugin.JpegImageFile 类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = processing_data(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        output = model(img)
    print(output)
    output = torch.argmax(output, dim=1)
    output = output.cpu().numpy()
    return inverted[output[0]]
# def test():
#     img=Image.open('./datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/train/00_00/00001.jpg')




# 输入图片路径和名称
image_path = './datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/val/00_00/00037.jpg'
# image_path = './datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/val/00_01/00010.jpg'

# 使用 opencv 读取图片
# image = cv2.imread(image_path)
image=Image.open(image_path)
# 打印返回结果
print(predict(image))
