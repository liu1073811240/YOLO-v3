# -*- coding:utf8 -*-

import os
import shutil


class BatchRename():
    '''
    批量重命名文件夹中的图片文件

    '''

    def __init__(self):
        self.path = r'D:\PycharmProjects(2)\YOLO v3\data2\images'  # 表示需要命名处理的文件夹
        self.output = r"D:\ACelebA\negative3"  # 表示移动到指定文件夹下

    def rename(self):
        filelist = os.listdir(self.path)  # 获取文件路径
        total_num = len(filelist)  # 获取文件长度（个数）
        i = 1  # 表示文件的命名是从1开始的
        for item in filelist:
            # if item.endswith('.jpg'):
            if item.endswith('.jpg') or item.endswith('.jpeg') or item.endswith('.png'):  # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                src = os.path.join(os.path.abspath(self.path), item)
                # dst = os.path.join(os.path.abspath(self.path), '' + str(i) + '.jpg')
                dst = os.path.join(os.path.abspath(self.path), str(i).zfill(2) + '.jpg')  # 处理后的格式也为jpg格式的，当然这里可以改成png格式
                # dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')   #  这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                try:
                    # 重命名文件名
                    os.rename(src, dst)

                    # 移动图片
                    # shutil.move(self.path + '/' + str(i).zfill(6) + '.jpg',
                    #             self.output + '/' + str(i).zfill(6) + '.jpg')

                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        # print('total %d to rename & converted %d jpgs' % (total_num, total_num-i+1))
        # print('total %d to move & converted %d jpgs' % (total_num, total_num-i+1))


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
