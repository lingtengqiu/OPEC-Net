from __future__ import absolute_import
import torch

import torch.nn.functional as F
import torch.nn as nn

def generate_2d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    accu_y = heatmaps.sum(dim=3)


    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)



    return accu_x, accu_y


def softmax_integral_tensor(preds, num_joints, hm_width, hm_height):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = F.softmax(preds, 2)
    score = torch.max(preds,-1)[0]

    # integrate heatmap into joint location
    x, y= generate_2d_integral_preds_tensor(preds, num_joints, hm_width, hm_height)
    x = x / float(hm_width) - 0.5
    y = y / float(hm_height) - 0.5


    preds = torch.cat((x, y), dim=2)
    preds*=2
    preds = preds.reshape((preds.shape[0], num_joints * 2))
    return preds,score
def make_conv3x3(
    in_channels,
    out_channels,
    dilation=1,
    stride=1,
    use_gn=False,
    use_relu=False,
    kaiming_init=True
):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False if use_gn else True
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    module = [conv,]
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


def make_fc(dim_in, hidden_dim, use_gn=False):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    '''
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc

class SELayer(nn.Module):
    '''
    squeeze and exc layer
    '''
    def __init__(self, in_planes, out_planes, reduction=16,use_gn = False):
        super(SELayer, self).__init__()
        self.use_gn = use_gn
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            make_fc(in_planes, out_planes // reduction,use_gn),
            nn.ReLU(inplace=True),
            make_fc(out_planes // reduction, out_planes,use_gn),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y
class Attention_layer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction):
        super(Attention_layer, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)

    def forward(self, x1, x2):
        """
        :param x1: low level feature
        :param x2: Hight level feature
        :return:  attention feature
        """
        fm = torch.cat([x1, x2], 1)
        channel_attetion = self.channel_attention(fm)
        fm = x1 * channel_attetion + x2
        return fm