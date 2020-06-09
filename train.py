import os
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box, compute_ts_road_map, compute_ats_bounding_boxes

# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
image_folder = '../data'
annotation_csv = '../data/annotation.csv'


unlabeled_scene_index = np.arange(106)


labeled_scene_index_train = np.arange(106, 127)
labeled_scene_index_val = np.arange(127, 134)
# In the submission the model is trained on full dataset.
labeled_scene_index_train = np.arange(106, 134)

# The labeled dataset can only be retrieved by sample.
# And all the returned data are tuple of tensors, since bounding boxes may have different size
# You can choose whether the loader returns the extra_info. It is optional. You don't have to use it.
transform = torchvision.transforms.ToTensor()

labeled_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index_train,
                                  transform=transform,
                                  extra_info=True
                                 )
trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)


labeled_valset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index_val,
                                  transform=transform,
                                  extra_info=True
                                 )
valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

def turn(xy, deg=-60):
    x, y = xy
    x -= 400
    y -= 400
    x, y = np.cos(deg / 180 * np.pi) * x - np.sin(deg / 180 * np.pi) * y, np.sin(deg / 180 * np.pi) * x + np.cos(deg / 180 * np.pi) * y
    x += 400
    y += 400
    return [x, y]

def turnN(array, n, deg=-60):
    array = list(array)
    for i in range(n):
        array = [turn(xy, deg) for xy in array]
    return np.array(array, dtype="float32")

import cv2

src_h = 128
src_w = 306
x = 525
src = np.float32([[0, src_h], [src_w-0, src_h], [0, 0], [src_w, 0]])
dst = [None] * 6
M = [None] * 6

# Right
dst[1] = np.float32([[410, 370], [410, 430], [400+x, 400-x * 0.56], [400+x, 400+x * 0.56]])
# Upper Right
dst[0] = turnN(dst[1], 1, -60)
# Upper Left
dst[3] = turnN(dst[1], 1, -120)
# Left
dst[4] = turnN(dst[1], 3)
# Lower Left
dst[5] = turnN(dst[1], 1, 120)
# Lower Right
dst[2] = turnN(dst[1], 1, 60)

M = [cv2.getPerspectiveTransform(src, d) for d in dst]

def getBirdsEye(img_batch):
    output = []
    for b in range(len(img_batch)):
        views = img_batch[b]
        res = []
        for direction in range(6):
            view = views[direction].permute(1, 2, 0).cpu().numpy()
            view_bottom = view[128:, :,]
            warped_img = cv2.warpPerspective(view_bottom, M[direction], (800, 800), flags=cv2.INTER_LINEAR)
            warped_img[warped_img == 0] = 'nan'
            res.append(warped_img)

        bird_eye = np.nanmean(res, axis=0)
        bird_eye = np.where(np.isnan(bird_eye), 0, bird_eye)
        output.append(torch.tensor(bird_eye).permute(2, 0, 1))
    return torch.stack(output)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


# Optional: you can use the script to pretrained the network on unlablled data

class Score:
    def __init__(self):
        self.sum = 0
        self.cnt = 0
        self.last_value = float('-inf')
    def update(self, value):
        self.sum += value
        self.cnt += 1
        self.last_value = value
    def average(self):
        return self.sum / self.cnt
    def last(self):
        return self.last_value

EPOCH_NUM = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

unet = ResNetUNet(1)
unet.to(DEVICE)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.0001)
print('[Using device: {}]'.format(DEVICE))


for epoch in range(EPOCH_NUM):
    print(f'--------------EPOCH {epoch}--------------')
    i = 0
    n_example = len(trainloader)
    # train
    t_score = Score()
    for sample, target, road_image, extra in trainloader:
        
        optimizer.zero_grad()
        x = getBirdsEye(sample)
        x = x.to(DEVICE)
        y = unet(x).squeeze(1)
        
        road_image = torch.stack(road_image).float().to(DEVICE)
        loss = criterion(y, road_image)
        # Threat Score: (on train data, not accurate)
        ts = compute_ts_road_map(y[0]>0, road_image[0])
        t_score.update(ts)
        
        if i % 100 == 0:
            print(f'\tTRAINING Batch: {i}/{n_example}, Threat Score: {t_score.last():.4}, Average Score: {t_score.average():.4}, Loss: {loss:.4}')

        loss.backward()
        optimizer.step()
        
        i += 1
    print(f'TRAINING Average Score: {t_score.average():.4}')
    
    # # val 
    # v_score = Score()
    # i = 0
    # n_example = len(valloader)
    # for sample, target, road_image, extra in valloader:
    #     unet.eval()
    #     with torch.no_grad():
    #         x = getBirdsEye(sample)
    #         x = x.to(DEVICE)
    #         y = unet(x).squeeze(1)

    #         road_image = torch.stack(road_image).float().to(DEVICE)
    #         # Threat Score: (on train data, not accurate)
    #         ts = compute_ts_road_map(y[0]>0, road_image[0])
    #         v_score.update(ts)
            
    #         if i % 100 == 0:
    #             print(f'\tVAL Batch: {i}/{n_example}, Threat Score: {v_score.last():.4}, Average Score: {v_score.average():.4}')
            
    #         i += 1
    # print(f'VAL Average Score: {v_score.average():.4}')

torch.save(unet.state_dict(), './saved_model_roadmap')

from models_yolo import *
from utils.parse_config import *
device = DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet('config/yolov3-tiny.cfg').to(device)
from utils.utils import *
model.apply(weights_init_normal)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def transform_bounding_box(target):
    res = []
    for b_i, target_batch in enumerate(target):
        boxes = target_batch['bounding_box']
        labels = target_batch['category']
        num_of_boxes = boxes.shape[0]  # boxes: N x 2 x 4
        output = torch.zeros((num_of_boxes, 6))
        for i, box in enumerate(boxes):
            tmp = torch.zeros(2, 4)
            tmp[0:] = 10 * box[0:] + 400
            tmp[1:] = -10 * box[1:] + 400
            x1 = (tmp[0][2] + tmp[0][3]) / 2 / 800
            x2 = (tmp[0][0] + tmp[0][1]) / 2 / 800
            y1 = (tmp[1][0] + tmp[1][2]) / 2 / 800
            y2 = (tmp[1][3] + tmp[1][1]) / 2 / 800
            output[i][0] = b_i
            output[i][2:6] = torch.Tensor([(x1+x2)/2, (y1+y2)/2, abs(x2-x1), abs(y2-y1)])
            output[i][1] = 1 if x1 < x2 else 0
            output = output.float()
        res.append(output)
    return torch.cat(res, 0)

def transform_back_bounding_box(outputs):
    res = []
    if outputs is None or len(outputs) == 0:
        return torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]).unsqueeze(0)
    for output in outputs:
        _x1, _y1, _x2, _y2 = output[:4] # in 800 x 800 scale
        turn = output[-1]
        x1, y1 = _x2, _y1
        x2, y2 = _x2, _y2
        x3, y3 = _x1, _y1
        x4, y4 = _x1, _y2
        if turn == 0:
            x1, x3 = x3, x1
            x2, x4 = x4, x2
        tmp = torch.tensor([[x1, x2, x3, x4], [y1, y2, y3, y4]])
        tmp[0] = (tmp[0] - 400) / 10
        tmp[1] = (tmp[1] - 400) / (-10)
        res.append(tmp)
    return torch.stack(res)

EPOCH_NUM = 5

model = Darknet('config/yolov3-tiny.cfg').to(device)
model.apply(weights_init_normal)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCH_NUM):
    print(f'--------------EPOCH {epoch}--------------')
    n_example = len(trainloader)
    # train
    t_score = Score()
    for i, (sample, target, road_image, extra) in enumerate(trainloader):
        
        optimizer.zero_grad()
        x = getBirdsEye(sample)
        x = x.to(DEVICE)
        
        targets = transform_bounding_box(target).to(DEVICE)
        loss, outputs = model(x, targets)
        
        loss.backward()
        optimizer.step()
        
        # handle outputs
        conf_thres = 0.2
        outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=0.3)
        num_of_boxes = 0 if outputs[0] is None else len(outputs[0])
        if i < 10: continue
        submit_output = [transform_back_bounding_box(outputs[0])]
        
        
        # Threat Score: (on train data, not accurate)
        ts = compute_ats_bounding_boxes(submit_output[0].cpu(), target[0]['bounding_box'])
        t_score.update(ts)
        
        if i % 50 == 0:
            print(f'\tTRAINING Batch: {i}/{n_example}, Threat Score: {t_score.last():.5f}, Average Score: {t_score.average():.5f}, Loss: {loss:.4}')

    print(f'TRAINING Average Score: {t_score.average():.4}')
    
#     # val Training with all data so validation is not accurate
#     v_score = Score()
#     n_example = len(valloader)
#     for i, (sample, target, road_image, extra) in enumerate(valloader):
#         model.eval()
#         with torch.no_grad():
#             x = getBirdsEye(sample)
#             x = x.to(DEVICE)

#             outputs = model(x)

#             # handle outputs
#             conf_thres = 0.2
#             outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=0.3)
#             num_of_boxes = 0 if outputs[0] is None else len(outputs[0])
#             submit_output = [transform_back_bounding_box(outputs[0])]


#             # Threat Score: (on train data, not accurate)
#             ts = compute_ats_bounding_boxes(submit_output[0].cpu(), target[0]['bounding_box'])
#             t_score.update(ts)

#             if i % 50 == 0:
#                 print(f'\tTRAINING Batch: {i}/{n_example}, Threat Score: {t_score.last():.5f}, Average Score: {t_score.average():.5f}')


    # print(f'VAL Average Score: {v_score.average():.4}')

torch.save(model.state_dict(), './saved_model_objdetect')
