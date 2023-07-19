import os.path

import paddle.io
from PIL import Image

import numpy as np


class caltech_dataset(paddle.io.Dataset):
    def __init__(self, data_path, mode="train"):
        """
        数据读取器
        :param data_path: 文件路径
        :param mode: 读取模式
        """
        super().__init__()
        self.data_path = data_path
        self.image_paths = []
        self.labels = []

        if mode == "train":
            with open(os.path.join(self.data_path, "train.txt"), "r", encoding="utf8") as f:
                self.info = f.readlines()
            for image_info in self.info:
                image_path, image_label = image_info.strip().strip("\t")
                self.image_paths.append(image_path)
                self.labels.append(image_label)
        else:
            pass

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((64, 64), Image.BILINEAR)
        image = np.array(image).astype("float32")
        image = image.transpose((2, 0, 1)) / 255
        label = np.array([label], dtype="int64")
        return image, label

    def __len__(self):
        return len(self.image_paths)
