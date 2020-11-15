# 配置文件
IMG_HEIGHT = 416
IMG_WIDTH = 416
CLASS_NUM = 4

"anchor box是对coco数据集聚类获得"
ANCHORS_GROUP_KMEANS = {  # k-means聚类得到的先验框（建议框）
    52: [[10, 13], [16, 30], [33, 23]],  # 在52尺寸特征图上聚类出来的框
    26: [[30, 61], [62, 45], [59, 119]],
    13: [[116, 90], [156, 198], [373, 326]]}

ANCHORS_GROUP = {  # 人工根据经验设置的框
    13: [[360, 360], [360, 180], [180, 360]],
    26: [[180, 180], [180, 90], [90, 180]],
    52: [[90, 90], [90, 45], [45, 90]]}

ANCHORS_GROUP_AREA = {  # 计算建议框的面积
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}

if __name__ == '__main__':

    for feature_size, anchors in ANCHORS_GROUP.items():
        print(feature_size)  # 13 26 52
        print(anchors)
    for feature_size, anchor_area in ANCHORS_GROUP_AREA.items():
        print(feature_size)
        print(anchor_area)
