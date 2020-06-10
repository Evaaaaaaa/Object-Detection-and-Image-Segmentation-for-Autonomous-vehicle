"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
# import ...
from model import Warping, ResNetUNet, CoordinateTransform
from models_yolo import *
from utils.parse_config import *
from utils.utils import *

# Put your transform function here, we will use it for our dataloader
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = ''
    round_number = 1
    team_member = ['Zhengyang Bian', 'Zhihan Li']
    contact_email = 'zb612@nyu.edu'

    def __init__(self, model_file1 = 'objdetect_v2_0506', model_file2='roadmap_v1_0427'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        self.model1 = Darknet('config/yolov3-tiny.cfg').cuda()
        self.model1.load_state_dict(torch.load(model_file1))
        self.coord = CoordinateTransform()

        self.model2 = ResNetUNet(1)
        self.model2.load_state_dict(torch.load(model_file2))
        self.model2.cuda()
        self.warp = Warping()

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        # Not implemented yet.
        x = self.warp.getBirdsEye(samples)
        x = x.cuda()
        outputs = self.model1(x)
        outputs = non_max_suppression(outputs, conf_thres=0.2, nms_thres=0.3)
        submit_output = self.coord.transform_back_bounding_box(outputs[0]).unsqueeze(0)

        return submit_output

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 

        x = self.warp.getBirdsEye(samples)
        x = x.cuda()
        y = self.model2(x).squeeze(1)
        
        return y > 0
