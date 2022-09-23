# -*- encoding: utf-8 -*-
'''
@file    : detect.py
@time    : 2022/09/22/22
@author  : PigGod666
@desc    : 加载pytorch训练好的模型并预测。
'''


import torch
import cv2 as cv
from module_file import OneModule, TwoModule
from utils.data_load import MyData


# 获取数据
# test_dataset = datasets.MNIST(root='/home/liu/D/d2l_data/images/MNIST', train=False, download=True, transform=transforms.ToTensor())
test_dataset = MyData("/home/liu/D/d2l_data/images/MNIST/MNIST_JPG/", False)
for i in test_dataset:
    # 加载模型
    net = TwoModule()
    # 加载模型参数
    net.load_state_dict(torch.load("./models/minst_two_9.pt"))
    # 设置模型为验证模式
    net.eval()
    # 预测，并将数据搬到cpu。
    ret = net(i[0].unsqueeze(0)).argmax(dim=1).cpu().item()
    # 将数据转为np的格式，准备用于可视化。
    img = (i[0].permute(1, 2, 0).numpy()).astype("uint8")
    # 可视化数据
    cv.imshow(str(ret), img)
    if cv.waitKey() == ord("q"):
        break
    cv.destroyAllWindows()
