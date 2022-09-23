# -*- encoding: utf-8 -*-
'''
@file    : module_file.py
@time    : 2022/09/22/23
@author  : PigGod666
@desc    : 该文件定义了两个分类模型
'''


from torch import nn
from torch.nn import Module, ModuleList
from torchsummary import summary


class BaseTwoConv(Module):
    # 将图片进行两次卷积再输出，此处不改变输入的形状。
    # 此处有使用分支，直接将输入与两次卷积的结果相加。
    def __init__(self, in_c, mid_c):
        super(BaseTwoConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_c),
            nn.LeakyReLU(),
            nn.Conv2d(mid_c, in_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU()
        )
        self.con_for_add = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU(),
        )

    def forward(self, imgs):
        in_imgs = imgs
        middle_img = self.block(imgs)
        return self.con_for_add(in_imgs + middle_img)


class BaseChangeChanel(Module):
    # 增大图片通道数、减小图片尺寸，再连接conv_num个BaseTwoConv
    def __init__(self, in_c, out_c, conv_num):
        super(BaseChangeChanel, self).__init__()
        self.change_chanel = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU()
        )
        self.moudle_li = ModuleList()
        for i in range(conv_num):
            self.moudle_li.append(BaseTwoConv(out_c, out_c*2))

    def forward(self, imgs):
        out = self.change_chanel(imgs)
        for i in self.moudle_li:
            out = i(out)
        return out
        

class OneModule(Module):
    # 使用一定数量的残差块加全连接层构成
    def __init__(self):
        super(OneModule, self).__init__()
        self.con = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.bcc1 = BaseChangeChanel(32, 64, 1)
        self.bcc2 = BaseChangeChanel(64, 128, 4)
        self.line = nn.Sequential(
            nn.Linear(7*7*128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, imgs):
        out = self.con(imgs)
        out = self.bcc1(out)
        out = self.bcc2(out)
        out = out.view(out.shape[0], -1)
        out = self.line(out)
        return out


class TwoModule(Module):
    def __init__(self):
        super(TwoModule, self).__init__()
        self.con = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.bcc1 = BaseChangeChanel(32, 64, 1)
        self.bcc2 = BaseChangeChanel(64, 128, 4)
        self.pool = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 10, kernel_size=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, imgs):
        out = self.con(imgs)
        out = self.bcc1(out)
        out = self.bcc2(out)
        out = self.pool(out)
        out = out.view(out.shape[0], -1)
        return out



if __name__ == '__main__':
    # net = OneModule()
    # net = net.to("cuda")
    # summary(net, (1, 28, 28))

    net_two = TwoModule()
    net_two = net_two.to("cuda")
    summary(net_two, (1, 28, 28))
