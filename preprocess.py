import json
import os
import zipfile

train_parameters = json.loads(open("configs/config.json", "r").read())


def unzip_data(src_path, target_path):
    """
    解压原始数据集，将src_path路径下的zip包解压至target_path目录下
    :param src_path:  zip包
    :param target_path:  目标目录
    :return:
    """
    if not os.path.isdir(target_path):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


def get_data_list(target_path, train_list_path, image_path):
    """
    生成数据列表
    :param target_path: 目标文件夹
    :param train_list_path: 训练列表
    :param image_path: 图片路径
    :return:
    """
    # 训练集图片路径
    train_data_paths = []
    # 训练集图片标签
    train_data_labels = []
    # 训练集文本
    train_data_list = open(target_path + "\\dataset\\train.txt", "r").readlines()
    # 所有图片数量
    all_class_images = 0
    # 标签类别
    class_detail = {}
    # 标签 json
    class_json = {}
    for train_data in train_data_list:
        train_data_path = image_path + train_data.split("\t")[0]
        train_data_label = train_data.split("\t")[1]


# unzip_data(train_parameters["src_path"], train_parameters["target_path"])
get_data_list(train_parameters["target_path"], train_parameters["train_list_path"], train_parameters["image_path"])
