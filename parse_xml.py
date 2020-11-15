from xml.dom.minidom import parse

xml_doc = r"./data/1.xml"

dom = parse(xml_doc)  # 开始解析xml文件

root = dom.documentElement

img_name = root.getElementsByTagName("path")[0].childNodes[0].data
# print(root.getElementsByTagName("path")[0].childNodes[0].data)  # C:\yolo_train_img\111.jpg

img_size = root.getElementsByTagName("size")[0]  # 图片大小内存地址

img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data  # 图片宽度：1080
img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data  # 图片高度：562
img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data  # 图片深度：3

objects = root.getElementsByTagName("object")

# for boxes in objects:
#     item = boxes.getElementsByTagName("item")
#     for box in item:
#         cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
#         x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
#         y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
#         x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
#         y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
#         print(cls_name,x1,y1,x2,y2)

item = root.getElementsByTagName("item")

for box in item:

    cls_name = box.getElementsByTagName("name")[0].childNodes[0].data  # 拿到name所对应的数据

    x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)  # 拿到x1的坐标
    y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
    x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
    y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
    print(cls_name, x1, y1, x2, y2)
