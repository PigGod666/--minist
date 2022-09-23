import torch
from torch import nn
from torchvision import datasets, transforms
import module_file
import time
from tqdm import tqdm
from utils.data_load import MyData
from torchvision.models import resnet50


def to_float(batch):
    data = torch.stack([img[0] for img in batch], 0)
    print([img[0] for img in batch][0].shape)
    print(data.size, data.max(), data.type)
    # print(data.shape, data.dtype)
    target = torch.Tensor([tar[1] for tar in batch])
    print(target.shape, target.dtype, target.dtype)
    return [data, target]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据准备
# train_dataset = datasets.MNIST(root='/home/liu/D/d2l_data/images/MNIST', train=True, download=True, transform=transforms.ToTensor())
# test_dataset = datasets.MNIST(root='/home/liu/D/d2l_data/images/MNIST', train=False, download=True, transform=transforms.ToTensor())
# train_load = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)
# test_load = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=True)

nTrainBatchSize = 128*3
trainDataset = MyData("/home/liu/D/d2l_data/images/MNIST/MNIST_JPG/", True)
testDataset = MyData("/home/liu/D/d2l_data/images/MNIST/MNIST_JPG/", False)
train_load = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=nTrainBatchSize, shuffle=True)
test_load = torch.utils.data.DataLoader(dataset=testDataset, batch_size=nTrainBatchSize, shuffle=True)

# for i in train_load:
#     print(i[0].shape, i[0].dtype, i[1].shape, i[1].dtype)
#     input(">>>")

# 读取模型
# net = module_file.TwoModule()
net = resnet50()

# 查看模型每一个层的名字和具体内容，方便对特定的层进行修改。
# for module in net.named_modules():
#     print(module)
# exit()

# resnet第一个层的输入是3通道的图片，minist数据是单通道的，此处对第一个层进行了修改。
cur_mod = getattr(net, "conv1")
setattr(net, "conv1", nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))

net = net.to(device)

# loss
loss = torch.nn.CrossEntropyLoss()

# 梯度下降算法
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


def train(train_data, test_data, optimizer, loss, epoch):
    for i in range(epoch):
        l_sum, train_acc, n = 0, 0, 0
        time_0 = time.time()
        # 训练
        for x, y in tqdm(train_data):
            # print("fkasjfd", x.shape, x.max(), x.dtype)
            x = x.to(device)
            y = y.to(device)
            # print(x.shape, x.dtype, y.shape, y.dtype)
            y_hat = net(x)
            # print(y_hat.shape, y.dtype)
            # input(">>>")
            # 计算loss
            l = loss(y_hat, y)
            # 梯度清零
            optimizer.zero_grad()
            # 计算梯度
            l.backward()
            # 反向传播，更新参数。
            optimizer.step()
            l_sum += l.cpu().item()
            train_acc += (y_hat.argmax(dim=1)==y).sum().cpu().item()
            n += y.shape[0]
        # 测试
        test_acc, test_n = 0, 0
        with torch.no_grad():
            for x, y in test_data:
                x = x.to(device)
                y = y.to(device)
                net.eval()      # 评估模式，关闭dropout，同时解决批归一化层输入图片数量不能为一的报错。
                test_acc += (net(x).argmax(dim=1)==y).sum().cpu().item()
                net.train()     # 训练模式。
                test_n += y.shape[0]
        if (i+1) % 10 == 0:
            torch.save(net.state_dict(), f"./models/minst_resnet50_{i}.pt")

        print(f"第{i+1}轮训练    损失：{l_sum:.3f}，训练精度：{train_acc/n:.3f}，测试精度：{test_acc/test_n:.3f}，时间：{time.time()-time_0:.3f}")


train(train_load, test_load, optimizer, loss, 10)
