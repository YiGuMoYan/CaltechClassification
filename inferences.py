import os

import numpy as np
import paddle
from PIL import Image

from model import caltech_model
from preprocess import train_parameters


def load_image(img_path):
    '''
    预测图片预处理
    '''
    image = Image.open(img_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((64, 64), Image.BILINEAR)
    image = np.array(image).astype('float32')
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = image / 255  # 像素值归一化
    return image


para_state_dict = paddle.load("./checkpoints/caltech_dataset_last")
model = caltech_model()
model.set_state_dict(para_state_dict)

infer_image_path = "/home/aistudio/data/dataset/dataset/test.txt"
infer_images = open(infer_image_path, 'r').readlines()

infer_image_list = []

for infer_image in infer_images:
    infer_image_list.append(load_image(os.path.join(train_parameters["image_path"], infer_image).replace("\n", "")))

# infer_image_list = np.array(infer_image_list)

outs = []

for i in range(len(infer_image_list)):
    dy_x_data = np.array(infer_image_list[i]).astype('float32')
    dy_x_data = dy_x_data[np.newaxis, :, :, :]
    img = paddle.to_tensor(dy_x_data)
    out = model(img)
    lab = np.argmax(out.numpy())  # argmax():返回最大数的索引
    image = infer_images[i].replace("\n", "")
    outs.append(f"{image}\t{lab}\n")

with open("result.txt", "w") as f:
    for out in outs:
        f.write(out)
print("结束")
