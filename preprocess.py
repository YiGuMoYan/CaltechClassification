import json
import os
import random
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


def get_data_list(target_path, train_list_path, image_path, class_list_path, readme_path):
    """
    生成数据列表
    :param target_path: 目标文件夹
    :param train_list_path: 训练列表
    :param image_path: 图片路径
    :param class_list_path: 标签文件路径
    :param readme_path: readme 文件路径
    :return:
    """
    # 训练集图片路径
    train_data_paths = []
    # 训练集图片标签
    train_data_labels = []
    # 训练集
    train_list = []
    # 所有图片数量
    all_class_images = 0
    # 标签类别
    class_detail = []
    # 训练集文本
    train_data_list = open(target_path + "dataset\\train.txt", "r").readlines()
    # 标签集
    class_data_list = open(class_list_path, "r").readlines()
    # 标签对应的图片数
    class_num = {}

    # 生成train.txt
    for train_data in train_data_list:
        train_data_path = image_path.replace("\\", "\\\\") + train_data.split("\t")[0]
        train_data_label = train_data.split("\t")[1].replace("\n", "")

        train_data_paths.append(train_data_path)
        train_data_labels.append(train_data_labels)

        train_list.append(f"{train_data_path}\t{train_data_label}\n")

        # 统计对应标签的训练集数量
        if train_data_label not in class_num.keys():
            class_num.setdefault(train_data_label, 0)
        class_num[train_data_label] += 1

    # 乱序
    random.shuffle(train_list)

    # 写入 train.txt
    with open(train_list_path, "w") as f:
        for train_data in train_list:
            f.write(train_data)

    # 生成标签集
    for class_data in class_data_list:
        class_name = class_data.split("\t")[0]
        class_label = class_data.split("\t")[1].replace("\n", "")
        # 标签类别默认内容
        class_detail_default = {"class_train_images": class_num[class_label], "class_label": int(class_label),
                                "class_name": class_name}
        class_detail.append(class_detail_default)

    all_class_images = len(train_data_paths)

    # 写入 readme
    readme_json = {"all_class_images": all_class_images, "class_detail": class_detail}
    with open(readme_path, 'w') as f:
        f.write(json.dumps(readme_json, sort_keys=True, indent=4, separators=(',', ': ')))

    print("数据生成完成")


if __name__ == "__main__":
    unzip_data(train_parameters["src_path"], train_parameters["target_path"])
    get_data_list(train_parameters["target_path"], train_parameters["train_list_path"], train_parameters["image_path"],
                  train_parameters["class_list_path"], train_parameters["readme_path"])
