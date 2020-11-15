import os
import time

from model import *
import cfg
import torch
import numpy as np
import PIL.Image as pimg
import PIL.ImageDraw as draw
from PIL import ImageFont
import tool
from torchvision import transforms
from Test_files.Convert_square import trans_square
from Test_files.padding_pixel import padding_pixel, padding_pixel2
import matplotlib.pyplot as plt


class Detector(torch.nn.Module):

    def __init__(self, save_path):
        super(Detector, self).__init__()

        self.net = MainNet().cuda()
        self.net.load_state_dict(torch.load(save_path))
        self.net.eval()

    # torch.randn(3, 3, 416, 416), 0.3, cfg.ANCHORS_GROUP
    def forward(self, input, thresh, anchors):
        output_13, output_26, output_52 = self.net(input)
        # print(output_13.shape)  # torch.Size([3, 27, 13, 13])

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])
        # 拿到所需的索引和输出向量(9个值)再解析出来。 32是反算到原图的比值

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)  # 按批次进行拼接

    def _filter(self, output, thresh):

        output = output.permute(0, 2, 3, 1)  # torch.Size([3, 13, 13, 27])

        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        # print(output.shape)  # torch.Size([3, 13, 13, 3, 9])

        # print(output[..., 0].shape)  # torch.Size([3, 13, 13, 3])
        mask = torch.sigmoid(output[..., 0]) > thresh  # 取到大于阈值的掩码 （iou）
        # print(mask.shape)  # torch.Size([3, 13, 13, 3])

        idxs = mask.nonzero()  # 取到非零元素的索引
        # print(idxs.shape)  # torch.Size([14, 4])

        vecs = output[mask]  # 利用掩码取选择输出的结果
        # print(np.shape(vecs))  # torch.Size([14, 9])

        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        if len(idxs) == 0:
            return torch.randn(0, 6).cuda()
        else:
            anchors = torch.tensor(anchors, dtype=torch.float32).cuda()
            # print(idxs.shape)  # torch.Size([14, 4])  N,H,W,3
            a = idxs[:, 3]  # 建议框:3
            # print(a.shape)  # torch.Size([14])

            # confidence = vecs[:, 0]
            # "压缩置信度值到0-1之间"
            confidence = torch.sigmoid(vecs[:, 0])
            # print(confidence.shape)  # torch.Size([14])

            _classify = vecs[:, 5:]
            # print(_classify.shape)  # torch.Size([14, 4])

            classify = torch.argmax(_classify, dim=1).float()
            # print(classify.shape)  # torch.Size([14])

            # idxs:N,H,W,3         网络所输出的vecs: iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h)
            cy = (idxs[:, 1].float() + torch.sigmoid(vecs[:, 2])) * t
            # print(cy.shape)  # torch.Size([14])

            cx = (idxs[:, 2].float() + torch.sigmoid(vecs[:, 1])) * t
            # print(cx.shape)  # torch.Size([14])

            w = anchors[a, 0] * torch.exp(vecs[:, 3])

            h = anchors[a, 1] * torch.exp(vecs[:, 4])
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = x1 + w
            y2 = y1 + h
            # print(confidence)
            out = torch.stack([confidence, x1, y1, x2, y2, classify], dim=1)

            return out


if __name__ == '__main__':
    save_path = "models/net_yolo.pth3"
    dicts = {"0.0": "人", "1.0": "老虎", "2.0": "狮子", "3.0": "熊猫"}
    font_path = "simsun.ttc"
    main_path = r"data2/images"
    font = ImageFont.truetype(font_path, size=20)
    detector = Detector(save_path)
    # a = torch.randn(3, 3, 416, 416).cuda()
    # y = detector(a, 0.3, cfg.ANCHORS_GROUP)
    # print(y.shape)
    # exit()
    count = 1
    for filename in os.listdir(main_path):
        # start_time = time.time()
        img1 = pimg.open(os.path.join(main_path, filename))

        w, h = img1.size
        merge_img, paste_coord = trans_square(img1)  # 将图片转成正方形

        w1, h1 = merge_img.size
        resize_img = merge_img.resize((416, 416))
        w2, h2 = resize_img.size
        scale = w2 / w1  # 缩放图片后的宽比上原图片的宽
        # print(scale)

        # img = np.array(img) / 255
        # img = torch.Tensor(img)
        # img = img.unsqueeze(0)
        # img = img.permute(0, 3, 1, 2)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = transform(resize_img)
        img = img.unsqueeze(0)

        # print(np.shape(img))  # torch.Size([1, 3, 416, 416])

        img = img.cuda()

        out_value = detector(img, 0.3, cfg.ANCHORS_GROUP)
        boxes = []
        # print(out_value.shape)  # torch.Size([42, 6])
        # print(out_value[..., -1].shape)  # torch.Size([42])

        for j in range(4):  # 几个类别循环几次
            classify_mask = (out_value[..., -1] == j)  # 类别相等才去输出相应框的掩码
            _boxes = out_value[classify_mask]
            _boxes = _boxes.cpu()
            boxes.append(tool.nms(_boxes))  # 同类别做NMS

        # for box in boxes:
        #     try:
        #         img_draw = draw.ImageDraw(img1)
        #         c, x1, y1, x2, y2 = box[0, 0:5]
        #         # print(c, x1, y1, x2, y2)
        #         img_draw.rectangle((x1, y1, x2, y2))
        #     except:
        #         continue

        # 遍历所有nms后的boxes, 三维

        for box in boxes:
            # 遍历每一类的所有box,二维
            # print(box)
            for _box in box:

                # try:
                img_draw = draw.ImageDraw(merge_img)

                confidence = round(_box[0].item(), 2)
                print(_box[1].item(), _box[2].item())  # 124, 58
                # x1 = _box[1].item() / scale
                # y1 = _box[2].item() / scale
                # x2 = _box[3].item() / scale
                # y2 = _box[4].item() / scale

                x1 = max(0, _box[1].item() / scale)
                y1 = max(0, _box[2].item() / scale)
                x2 = min(w1, _box[3].item() / scale)
                y2 = min(h1, _box[4].item() / scale)
                cls = _box[5].item()
                cls = dicts[str(cls)]  # 拿到字典所对应的字符串

                print(cls, confidence, x1, y1, x2, y2)

                img_draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)

                padding_pixel(merge_img, x1, y1, 80, 20)
                img_draw.text((x1, y1-20), cls, fill=(0, 0, 0), font=font)
                img_draw.text((x1+40, y1-20), str(confidence), fill=(0, 0, 0), font=font)

                # except:
                #     continue
        merge_img2 = merge_img.crop((paste_coord[0], paste_coord[1], paste_coord[0]+w, paste_coord[1]+h))
        # merge_img2.save("./Save_images2/{}.jpg".format(count))
        count += 1

        # end_time = time.time()
        # print("侦测一张图片所用时间", start_time-start_time)  # 0.8s
        plt.imshow(merge_img2)

        plt.pause(1)



