# -*- encoding: utf-8 -*-
'''
@file    : onnx_to_trt.py
@time    : 2022/09/22/23
@author  : PigGod666
@desc    : 该文件用于把onnx文件转换为tensorrt的引擎文件，并使用tensorrt预测。
'''


from time import time
from cuda import cudart
import cv2
import numpy as np
import os
import tensorrt as trt
import torch as t
import torch.nn.functional as F
import torch
import os
from pathlib import Path

import utils.calibrator as calibrator

base_path = Path(__file__).parent.absolute()
np.random.seed(97)
t.manual_seed(97)
t.cuda.manual_seed_all(97)
t.backends.cudnn.deterministic = True
nHeight = 64
nWidth = 64
onnxFile = os.path.join(base_path, "models/minst_resnet50_9.onnx")
trtFile = os.path.join(base_path, "models/minst_resnet50_9_fp16.plan")
dataPath = "/home/liu/D/d2l_data/images/MNIST/MNIST_JPG/"
inferenceImage = os.path.join(dataPath, "test/00021-6.0.jpg")

# for FP16 mode
isFP16Mode = True
# for INT8 model
isINT8Mode = False
nCalibration = 1
cacheFile = "./int8.cache"
calibrationDataPath = dataPath + "test/"

# os.system("rm -rf ./*.onnx ./*.plan ./*.cache")
np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()


# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
if isFP16Mode:
    config.flags = 1 << int(trt.BuilderFlag.FP16)
if isINT8Mode:
    config.flags = 1 << int(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, nHeight, nWidth), cacheFile)

if False:
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, (1, 1, nHeight, nWidth), (4, 1, nHeight, nWidth), (8, 1, nHeight, nWidth))
    config.add_optimization_profile(profile)

    # network.unmark_output(network.get_output(0))  # 去掉输出张量 "y"
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")

    # engine保存为文件
    with open(trtFile, "wb") as f:
        f.write(engineString)
else:
    with open(trtFile, "rb") as f:
        engineString = f.read()

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

context = engine.create_execution_context()
context.set_binding_shape(0, [1, 1, nHeight, nWidth])
#print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
#for i in range(engine.num_bindings):
#    print("Bind[%2d]:i[%d]->"%(i,i) if engine.binding_is_input(i) else "Bind[%2d]:o[%d]->"%(i,i-nInput),
#            engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i),engine.get_binding_name(i))


def inference(engine, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    data = img.astype(np.float32).reshape(1, 1, nHeight, nWidth)
    bufferH = []
    bufferH.append(data)
    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    # print("inputH0 :", bufferH[0].shape)
    # print("outputH0:", bufferH[-1].shape)
    # print(bufferH[-1])
    ret = bufferH[-1]
    # print(t.argmax(F.softmax(torch.tensor(ret), dim=1), dim=1))

    for buffer in bufferD:
        cudart.cudaFree(buffer)

    # 结果可视化。
    cv2.imshow(f"{int(t.argmax(F.softmax(torch.tensor(ret), dim=1), dim=1)[0].numpy())}", img)
    key = cv2.waitKey()
    cv2.destroyAllWindows()
    if key == ord("q"):
        exit()
    return ret

t0 = time()
for path_p in (Path(dataPath) / "test").glob("*.jpg"):
    ret = inference(engine, str(path_p))
    ret = int(t.argmax(F.softmax(torch.tensor(ret), dim=1), dim=1)[0].numpy())
    # print("ret: ", ret,  str(path_p)[-7])
    # break
t2 = time()
print(t2-t0, (t2-t0)/10000)