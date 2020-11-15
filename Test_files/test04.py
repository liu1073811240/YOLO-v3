import torch
import torch.nn as nn
import math
import torchvision

# entroy = nn.CrossEntropyLoss()
# input = torch.Tensor([[-0.7715, -0.6205, -0.2562]])
# target = torch.tensor([0])
# print(input.shape)
# print(target.shape)
#
# output = entroy(input, target)
# print(output)
# 根据公式计算


# 查看torch的版本
print(torch.__version__)
# 查看torchvision的版本
print(torchvision.__version__)

# 查看cuda是否安装成功，成功为True，不成功为False
print(torch.cuda.is_available())
# 查看cudnn是否安装成功，成功为True，不成功为False
print(torch.cudnn_is_acceptable(torch.Tensor(1).cuda()))

# 查看cuda的版本
print(torch.version.cuda)
# 查看cudnn的版本
print(torch.backends.cudnn.version())
# 查看当前GPU
print(torch.cuda.get_device_name(0))
