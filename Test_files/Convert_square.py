from PIL import Image
import os


def trans_square(image):  # 将图片转成正方形
    r"""Open the image using PIL."""
    image = image.convert('RGB')
    w, h = image.size
    background = Image.new('RGB', size=(max(w, h), max(w, h)), color=(0, 0, 0))  # 创建背景图，颜色值为127
    length = int(abs(w - h) // 2)  # 一侧需要填充的长度
    box = (length, 0) if w < h else (0, length)  # 粘贴的位置
    background.paste(image, box)

    return background, box

# img = Image.open(r"D:\PycharmProjects(2)\YOLO v3\data2\images\01.jpg")
# trans_square(img)
# if __name__ == '__main__':
#     main_path = r"D:\PycharmProjects(2)\YOLO v3\data2\images"
#     count = 1
#     for filename in os.listdir(main_path):
#         img = Image.open(os.path.join(main_path, filename))
#         background = trans_square(img)
#
#         background = background.resize((416, 416))
#         background.save(r"D:\PycharmProjects(2)\YOLO v3\data2\images2\{0}.jpg".format(str(count).zfill(2)))
#
#         count += 1






