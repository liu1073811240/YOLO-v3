from xml.dom.minidom import parse
import os
import traceback
from PIL import Image

dir_path = r"D:\PycharmProjects(2)\YOLO v3\data2"
xml_path = r"D:\PycharmProjects(2)\YOLO v3\data2\outputs2"

label_filename = os.path.join(dir_path, "label.txt")
dicts = {"人": 0, "老虎": 1, "狮子": 2, "熊猫": 3}

try:
    label_file = open(label_filename, "w")

    count = 0
    for filename in os.listdir(xml_path):
        try:
            dom = parse(os.path.join(xml_path, filename))  # 开始解析xml文件
            root = dom.documentElement

            img_name = root.getElementsByTagName("path")[0].childNodes[0].data  # D:\PycharmProjects(2)\YOLO v3\data2\images\01.jpg

            item = root.getElementsByTagName("item")

            label_file.write("images2/{0}.jpg ".format(str(count+1).zfill(2)))
            for box in item:

                cls_name = box.getElementsByTagName("name")[0].childNodes[0].data  # 拿到name所对应的数据
                value = dicts[cls_name]

                x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)  # 拿到x1的坐标
                y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
                x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
                y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
                cx = int(x1 + (x2 - x1) / 2)
                cy = int(y1 + (y2 - y1) / 2)
                w = x2 - x1
                h = y2 - y1

                label_file.write("{0} {1} {2} {3} {4} " .format(
                    value, cx, cy, w, h
                ))

            label_file.write("\n")

            count += 1
        except Exception as e:
            traceback.print_exc()

finally:
    label_file.close()