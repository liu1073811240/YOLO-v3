import dataset
from model import *
import torch
from torch.utils.data import DataLoader
import os


# 损失
def loss_fn(output, target, alpha):
    conf_loss_fn = torch.nn.BCEWithLogitsLoss()
    coord_loss_fn = torch.nn.MSELoss()
    cls_loss_fn = torch.nn.CrossEntropyLoss()

    # [N,C,H,W]-->>[N,H,W,C]
    output = output.permute(0, 2, 3, 1)
    # [N,C,H,W]-->>[N,H,W,3,15]
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    output = output.cpu().double()

    # print(target.shape)  # torch.Size([2, 13, 13, 3, 9])
    mask_obj = target[..., 0] > 0  # 取大于零的掩码来选择输出和标签的值.  （iou值大于零，把背景过滤掉）
    # print(mask_obj.shape)  # torch.Size([2, 13, 13, 3])
    output_obj = output[mask_obj]
    # print(output.shape)  # torch.Size([2, 13, 13, 3, 9])
    # print(output_obj.shape)  # torch.Size([9, 9])
    target_obj = target[mask_obj]
    # print(target_obj.shape)  # torch.Size([9, 9])
    # print(output_obj[:, 0].shape)  # torch.Size([9])
    # print(target_obj[:, 0].shape)  # torch.Size([9])
    # print(output_obj[:, 1:5].shape)  # torch.Size([9, 4])
    # print(target_obj[:, 1:5].shape)  # torch.Size([9, 4])
    # print(output_obj[:, 5:].shape)  # torch.Size([9, 4])
    # print(target_obj[:, 5:].shape)  # torch.Size([9, 4])

    loss_obj_conf = conf_loss_fn(output_obj[:, 0], target_obj[:, 0])
    loss_obj_coord = coord_loss_fn(output_obj[:, 1:5], target_obj[:, 1:5])

    target_obj = torch.argmax(target_obj[:, 5:], dim=1)
    loss_obj_cls = cls_loss_fn(output_obj[:, 5:], target_obj)

    loss_obj = loss_obj_conf + loss_obj_coord + loss_obj_cls

    mask_noobj = target[..., 0] == 0  # 没有目标的损失函数，只需要训练置信度。
    output_noobj = output[mask_noobj]
    # print(output_noobj.shape)  # torch.Size([1008, 9])

    target_noobj = target[mask_noobj]
    # print(target_noobj.shape)  # torch.Size([1008, 9])
    # print(output_noobj[:, 0].shape)  # torch.Size([1008])
    # print(target_noobj[:, 0].shape)  # torch.Size([1008])

    loss_noobj = conf_loss_fn(output_noobj[:, 0], target_noobj[:, 0])
    loss = alpha * loss_obj + (1 - alpha) * loss_noobj  # 正样本训练的比较多，召回率低

    return loss


if __name__ == '__main__':
    save_path = "models/net_yolo.pth3"
    myDataset = dataset.MyDataset()
    train_loader = DataLoader(myDataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MainNet().to(device)

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("NO Param")

    net.train()
    opt = torch.optim.Adam(net.parameters())

    epoch = 0
    while True:
        for target_13, target_26, target_52, img_data in train_loader:
            # print(target_13.shape)  # torch.Size([2, 13, 13, 3, 9])

            img_data = img_data.to(device)
            output_13, output_26, output_52 = net(img_data)
            # print(output_13.shape)  # torch.Size([2, 45, 13, 13])

            loss_13 = loss_fn(output_13, target_13, 0.9)
            loss_26 = loss_fn(output_26, target_26, 0.9)
            loss_52 = loss_fn(output_52, target_52, 0.9)
            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % 10 == 0:
                torch.save(net.state_dict(), save_path)

                print('save epoch: {}'.format(epoch))

            print("loss:", loss.item())

        epoch += 1
