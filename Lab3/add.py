data_path = './datasets/5f680a696ec9b83bb0037081-momodel/data/image'
path="MFRD/AFDB_face_dataset"

import os
import glob
import random
import cv2

Image_name_list=[]
# data_dir = 'dataset/'  # 文件地址/名称
classes = os.listdir(path)
for cls in classes:
    # print(cls)
    Image_glob = os.path.join(path+"/"+cls, "*.png")
    Image_name_list.extend(glob.glob(Image_glob))
    Image_glob = os.path.join(path+"/"+cls, "*.jpg")
    Image_name_list.extend(glob.glob(Image_glob))

print(len(Image_name_list))
random.shuffle(Image_name_list)
index=0
for file in Image_name_list:
    img = cv2.imread(file)
    # print(img.shape)
    # print(img)
    index+=1
    if(index>3000):
        break
    # img = cv2.resize(img, (160, 160))
    # print("/datasets/5f680a696ec9b83bb0037081-momodel/data/image/mask/"+str(index)+".jpg")
    cv2.imwrite("datasets/5f680a696ec9b83bb0037081-momodel/data/image/nomask/"+str(index)+".jpg", img)
    # break
    print(index)
