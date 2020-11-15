from PIL import ImageDraw, Image


def padding_pixel(img, x1, y1, width, height):
    draw = ImageDraw.ImageDraw(img)
    for i in range(width):
        for j in range(height):
            draw.point((x1 + i, y1 + j - 20), fill=(255, 0, 0))


def padding_pixel2(img, x1, y1, width, height):
    draw = ImageDraw.ImageDraw(img)
    for i in range(width):
        for j in range(height):
            draw.point((x1 + i, y1 + j), fill=(255, 0, 0))


if __name__ == '__main__':
    img = Image.open(r"D:\PycharmProjects(2)\YOLO v3\Test_files\dog-cycle-car.png")
    padding_pixel(img, 240, 60, 100, 100)
    img.show()