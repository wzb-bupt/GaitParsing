#!/usr/bin/env python
# coding=utf-8

import functools
import glob
import os
import os.path as osp
import time
import math
import cv2
import numpy as np
import torch
import tqdm
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

from u2net import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

get_class_colors = [[0, 0, 0], [255, 0, 0], [255, 255, 0], [0, 0, 255],
                    [255, 0, 255], [0, 255, 0], [0, 255, 255], [80, 0, 71],
                    [49, 0, 74], [125, 0, 34], [126, 0, 67]]

get_class_names = [
    'background', 'head', 'body', 'r_arm', 'l_arm', 'r_leg', 'l_leg'
]


class PedParsing:
    def __init__(self, model_path, gpu_id=0):
        self.input_height = 144
        self.input_width = 96
        self.transform = transforms.Normalize((0.5,0.5,0.5),(0.5, 0.5, 0.5))
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.num_classes = len(get_class_names)
        self.palette_idx = np.array(get_class_colors)


    def loadModel(self):
        self.model = U2NET(3, self.num_classes)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        assert(torch.cuda.is_available())   
        self.model.cuda(self.gpu_id)

    def parsingPed(self, inputImages): 
        batch_imgs = []
        batch_parsing = []
        for i in range(len(inputImages)):
            img = inputImages[i]
            crop_img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            crop_img = torch.from_numpy(crop_img.transpose((2, 0, 1)))        
            crop_img = self.transform(crop_img.float().div(255.0))
            crop_img = crop_img.view(1, crop_img.shape[0],crop_img.shape[1],crop_img.shape[2])            
            if i == 0:
               batch_imgs = crop_img
            else:
               batch_imgs = torch.cat((batch_imgs, crop_img),0)

        outputs, _, _, _, _, _, _ = self.model(Variable(batch_imgs.data.cuda(self.gpu_id)))
        outputs = outputs.cpu()
        
        for i in range(len(inputImages)):
            img = inputImages[i]
            img_height, img_width = img.shape[:2]
            prediction = outputs[i,:,:,:].data.numpy()
            result = np.zeros((self.input_height, self.input_width, 3))
            for i in range(prediction.shape[1]):
                for j in range(prediction.shape[2]):
                    result[i][j] = self.palette_idx[np.argmax(prediction[:, i, j])]

            parsing_im = cv2.resize(result, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            batch_parsing.append(parsing_im)

        return batch_parsing

def main(INPUT_PATH, OUTPUT_PATH, _id, _type, _view, batch_size):
    seq_path = osp.join(INPUT_PATH, _id, _type, _view)
    save_path = osp.join(OUTPUT_PATH, _id, _type, _view)
    image_name_list = os.listdir(seq_path)
    image_path_list = []
    parsing_all = []
    im_num = len(image_name_list)
    for i in range(im_num):
        image_path_list.append(osp.join(seq_path, image_name_list[i]))

    batch_nums = math.ceil(im_num/batch_size)
    for b in range(batch_nums):
        input_imgs = []
        for i in range(batch_size):
            if b*batch_size+i >= im_num:
                break
            image_path = image_path_list[b*batch_size+i]
            img = cv2.imread(image_path)
            input_imgs.append(img)
        parsing_all = parsing_all + pp.parsingPed(input_imgs)
    
    for i in range(im_num):
        print('processing:', save_path, i)
        image_name = image_name_list[i]
        os.makedirs(save_path, exist_ok=True)
        save_image_path = os.path.join(save_path, image_name.replace('.jpg', '.png'))
        cv2.imwrite(save_image_path, parsing_all[i])

if __name__ == '__main__':
    model_path = 'parsing_u2net.pth'
    pp = PedParsing(model_path)
    pp.loadModel()

    INPUT_PATH = "<path of your dataset>"
    OUTPUT_PATH = "<the output path>"
    batch_size = 16
    for _id in sorted(os.listdir(INPUT_PATH)):
        for _type in sorted(os.listdir(osp.join(INPUT_PATH, _id))):
            for _view in sorted(os.listdir(osp.join(INPUT_PATH, _id, _type))):
                main(INPUT_PATH, OUTPUT_PATH, _id, _type, _view, batch_size)
                # print('Processing:', osp.join(INPUT_PATH, _id, _type, _view), ' Done!')

    
