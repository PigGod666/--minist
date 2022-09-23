# -*- encoding: utf-8 -*-
'''
@file    : data_load.py
@time    : 2022/09/21/22
@author  : PigGod666
@desc    : pytorch加载自己的数据集 必须要重写__getitem__和__len__方法
'''


import torch
import cv2
from glob import glob
import numpy as np


class MyData(torch.utils.data.Dataset):

    def __init__(self, data_folder, isTrain=True):
        if isTrain:
            self.data = sorted(glob(data_folder + "train/*.jpg"))
        else:
            self.data = sorted(glob(data_folder + "test/*.jpg"))

    def __getitem__(self, index):
        imageName = self.data[index]
        data = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
        data = data.transpose(2, 0, 1)
        label = np.array(imageName[-7], dtype=np.int64)
        return torch.from_numpy(data.astype(np.float32)), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
