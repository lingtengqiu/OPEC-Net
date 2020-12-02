from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from engineer.models.common.helper import *

class HM_Extrect(nn.Module):

    def __init__(self,out_channel):
        super(HM_Extrect,self).__init__()


        self.level_conv1_1 = make_conv3x3(512,256)
        self.level_conv2_1 = make_conv3x3(256,128)
        self.level_conv3_1 = make_conv3x3(128,128)
        self.level_conv_out = make_conv3x3(128, out_channel)
        self.level_conv1_up = nn.ConvTranspose2d(256, 256, 2, 2, 0)
        self.level_conv2_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.attention_conv1_up_a = Attention_layer(512,256,reduction = 1)
        self.attention_conv2_up_a = Attention_layer(256,128,reduction = 1)

    def forward(self,features):
        results = []

        x = F.relu(self.level_conv1_1(features[0]))
        results.append(x)
        x = F.relu(self.level_conv1_up(x))
        x = self.attention_conv1_up_a(x,features[1])
        x = (F.relu(self.level_conv2_1(x)))
        results.append(x)

        x = F.relu(self.level_conv2_up(x))
        x = self.attention_conv2_up_a(x,features[2])
        x = (F.relu(self.level_conv3_1(x)))
        results.append(x)
        return results,self.level_conv_out(x)