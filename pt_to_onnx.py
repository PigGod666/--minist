# -*- encoding: utf-8 -*-
'''
@file    : pt_to_onnx.py
@time    : 2022/09/22/23
@author  : PigGod666
@desc    : 将pytorch训练好的模型转为onnx，并使onnx支持动态输入。
'''


import torch
from torch._C import device
from module_file import TwoModule
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision.models import resnet50
from torch import nn

# 获取模型
# net = TwoModule()
net = resnet50()
# resnet第一个层的输入是3通道的图片，minist数据是但通道的，此处对第一个层进行了修改。
setattr(net, "conv1", nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))

# 加载权重文件。
net.load_state_dict(torch.load("models/minst_resnet50_9.pt"))
net = net.to(device)

# 准备一个输入数据
input = torch.randn(1, 1, 64, 64)
input = input.to(device)

# torch.onnx.export(net, input, "./models/minst_two_9.onnx", verbose=True)

# pytorch转onnx，并支持动态输入。
torch.onnx.export(net, 
                  input, 
                  "./models/minst_resnet50_9.onnx", 
                  input_names=["x"], 
                  output_names=["y"], 
                  do_constant_folding=True, 
                  verbose=True, 
                  keep_initializers_as_inputs=True, 
                  opset_version=12, 
                  dynamic_axes={"x": {0: "nBatchSize", 2 : 'in_width', 3: 'int_height'}, "y": {0: "nBatchSize", 2 : 'in_width', 3: 'int_height'}}
                  )
