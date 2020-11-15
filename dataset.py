import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cfg
import os
from PIL import Image
import math
from torchvision import transforms

LABEL_FILE_PATH = "data2/label.txt"
IMG_BASE_DIR = "data2"
# LABEL_FILE_PATH = "data/person_label.txt"
# IMG_BASE_DIR = "data"

transforms = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def one_hot(cls_num, i):
    b = np.zeros(cls_num)
    b[i] = 1.
    return b


class MyDataset(Dataset):

    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()  # 读取所有行数据

    def __len__(self):
        return len(self.dataset)  # 返回数据集的长度

    def __getitem__(self, index):
        labels = {}
        line = self.dataset[index]  # 拿到整行数据，比如： images/21.jpg 0 18 45 258 264 1 258 99 290 250

        strs = line.strip().split()  # ['images/25.jpg', '3', '74', '142', '357', '323']
        # print(strs)

        _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))  # 打开每一张图片得到图片数据
        img_data = transforms(_img_data)

        # _boxes = np.array(float(x) for x in strs[1:])
        # 拿到图片数据
        _boxes = np.array(list(map(float, strs[1:])))  # [0.0, 2.0, 49.0, 344.0, 261.0, 1.0, 103.0, 76.0, 496.0, 303.0]
        # print(_boxes)

        # 拿到标签框信息
        boxes = np.split(_boxes, len(_boxes) // 5)
        # [array([  0.,   2.,  49., 344., 261.]), array([  1., 103.,  76., 496., 303.])]

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():  # 人工设置的建议框
            # print(feature_size)  # 13
            # print(anchors)  # [[360, 360], [360, 180], [180, 360]]
            # print(cfg.CLASS_NUM)

            # 生成13尺寸、26尺寸、52尺寸的零矩阵，目的是把有目标的中心0替换成1
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))
            # 3表示每组尺寸有三个建议框
            # print(labels)

            for box in boxes:  # 遍历每个目标的标签框
                cls, cx, cy, w, h = box  # 1.0 256.0 308.0 513.0 617.0
                # print(cls, cx, cy, w, h)

                # 目标中心点取到小数部分和整数部分， 网络学习的是小数部分。
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)  # 相当于cx / 32
                # print(feature_size,'---- ',cy,cy*feature_size)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)

                for i, anchor in enumerate(anchors):  # 循环一个尺度下的三个建议框。
                    # print(i)  # 0
                    # print(anchor)  # [360, 360]

                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]  # 循环三个建议框的面积
                    # print(anchor_area)  # 129600

                    p_w, p_h = w / anchor[0], h / anchor[1]  # 标签框（真实框）的宽度除以建议框的宽度
                    p_area = w * h  # 标签框的面积

                    # 值相当于置信度. 建议框和真实框都是同一个中心点，要求是同心框。 作用可以过滤掉一些比较小的建议框
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    # print(iou)
                    # print(*one_hot(cfg.CLASS_NUM, int(cls)))  # 0.0 0.0 0.0 1.0

                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h),
                         *one_hot(cfg.CLASS_NUM, int(cls))])  # 10,i
                    # print(labels)  # 前面把H,W,3作为维度，后面15个值作为填充

        # print(labels[13].shape)  # (13, 13, 3, 9)
        # print(labels[26].shape)  # (26, 26, 3, 9)
        # print(labels[52].shape)  # (52, 52, 3, 9)
        return labels[13], labels[26], labels[52], img_data


if __name__ == '__main__':

    # x = one_hot(10, 2)
    # print(x)

    data = MyDataset()
    dataloader = DataLoader(data, 2, shuffle=True)
    for i, x in enumerate(dataloader):
        print("====")
        # print(x[0].shape)  # torch.Size([2, 13, 13, 3, 9])
        # print(x[1].shape)  # torch.Size([2, 26, 26, 3, 9])
        # print(x[2].shape)  # torch.Size([2, 52, 52, 3, 9])
        # print(x[3].shape)  # torch.Size([2, 3, 416, 416])
    # for target_13, target_26, target_52, img_data in dataloader:
        # print(target_13.shape)  # torch.Size([2, 13, 13, 3, 9])
        # print(target_26.shape)  # torch.Size([2, 26, 26, 3, 9])
        # print(target_52.shape)  # torch.Size([2, 52, 52, 3, 9])
        # print(img_data.shape)  # torch.Size([2, 3, 416, 416])
