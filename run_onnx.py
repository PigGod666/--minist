# -*-coding: utf-8 -*-
 
import os, sys
from pathlib import Path
from time import time
import cv2
import torch.nn as nn
import torch
sys.path.append(os.getcwd())
import onnxruntime
import numpy as np
 
 
class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))
 
    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed
 
    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        ret = self.onnx_session.run(None, {self.input_name[0]: image_tensor})
        # ret = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        # input_feed = self.get_input_feed(self.input_name, image_tensor)
        # ret = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return ret


def ret_softmax(ret):
    ret = torch.tensor(ret)
    soft_0 = nn.Softmax(dim=1)
    ret = soft_0(ret)
    # print("ret max", torch.argmax(ret, dim=1))
    return torch.argmax(ret, dim=1).cpu().numpy()

if __name__ == '__main__':
    model_onnx = ONNXModel("/home/liu/E/python/d2l/class/MNIST/models/minst_resnet50_9.onnx")
    t0 = time()
    for img_path in Path("/home/liu/D/d2l_data/images/MNIST/MNIST_JPG/test").glob("*.jpg"):
        img_s = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        # img_s = cv2.resize(img_s, (28, 28))
        img = img_s.astype(np.float32)
        # img = img/255.0
        img = img[:, :, np.newaxis]

        img = img.transpose((2, 0, 1))              # (3, 96, 96)
        img = img[np.newaxis,:,:,:]
        # print(img.shape)
        # t0 = time()
        ret = model_onnx.forward(img)
        # print(ret)
        # print(time() - t0)
        num = ret_softmax(ret[0])[0]
        # print(len(ret), ret[0].shape, type(ret[0]))
        # print(num, str(img_path)[-7])
        # cv2.imshow(f"{num}", img_s)
        # if cv2.waitKey() == ord("q"):
        #     break
        # cv2.destroyAllWindows()
    t2 = time()
    print(t2-t0, (t2-t0)/10000)