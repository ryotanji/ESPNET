import numpy as np
import torch
from torch.autograd import Variable
import glob
import cv2
import os
import time
from argparse import ArgumentParser
import math
import Model as Net
​
import cv2
import pickle
import numpy as np
​
def evaluateModel(modelType,tgtimg,up,model,gpu=True):
​
    # gloabl mean and std values
    mean = [ 99.68974, 108.78835, 106.222084]
    std = [64.62489, 63.845295, 61.75027]
​
    img = tgtimg
​
    img = img.astype(np.float32)
    for j in range(3):
        img[:, :, j] -= mean[j]
    for j in range(3):
        img[:, :, j] /= std[j]
​
    # resize the image to 1024x512x3
    img = cv2.resize(img, (640, 480))
    # normalize
    img /= 255
    img = img.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
    with torch.no_grad():
        img_variable = Variable(img_tensor)
        if gpu:
            img_variable = img_variable.cuda()
        img_out = model(img_variable)
​
        if modelType == 2:
            img_out = up(img_out)
​
    classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()
​
    return classMap_numpy
​
def Evaluate(weightsDir='./weights',modelType=1,classes=2, gpu=True):
    up = None
    if modelType == 2:
        up = torch.nn.Upsample(scale_factor=8, mode='bilinear')
        if gpu:
            up = up.cuda()
​
    p = 2
    q = 8
    if modelType == 2:
        modelA = Net.ESPNet_Encoder(classes, p, q)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        model_weight_file = weightsDir + os.sep + 'encoder' + os.sep  + 'model_300.pth'
        if not os.path.isfile(model_weight_file):
            print('Pre-trained model file does not exist. Please check ../pretrained/encoder folder')
            exit(-1)
        modelA.load_state_dict(torch.load(model_weight_file))
    elif modelType == 1:
        modelA = Net.ESPNet(classes, p, q)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        model_weight_file = weightsDir + os.sep + 'decoder' + os.sep  + 'model_300.pth'
        if not os.path.isfile(model_weight_file):
            print('Pre-trained model file does not exist. Please check ../pretrained/decoder folder')
            exit(-1)
        modelA.load_state_dict(torch.load(model_weight_file))
​
    else:
        print('Model not supported')
​
    if gpu:
        modelA = modelA.cuda()
​
    # set to evaluation mode
    modelA.eval()
​
    return modelA,up
