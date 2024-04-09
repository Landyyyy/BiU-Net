# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from .MSF import ChannelTransformer
import .SSA as SSA

from timm.models import create_model

import Config
import math


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class UpDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        x = torch.cat([skip_x, up], dim=1)
        return self.nConvs(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class eca_block(nn.Module):
    def __init__(self, in_channel, b=1, gama=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size + 1
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view([b, 1, c])
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x.view([b, c, 1, 1])
        outputs = self.relu(x * inputs)
        return outputs


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()
        self.W = Conv(ch_1 + ch_2, ch_out, 3, bn=True, relu=True)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        g = self.eca1(g)
        x = self.eca2(x)
        fuse = self.W(torch.cat([g, x], 1))
        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class SingleConvLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(SingleConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.residual = residual
        tmp_ch = out_channels // 2
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, tmp_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(tmp_ch),
            nn.ReLU(inplace=True)
        )
        self.local_conv = SingleConvLayer(tmp_ch, tmp_ch // 2, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                                          bias=True)
        self.conv2 = SingleConvLayer(tmp_ch, tmp_ch // 2, 3, 1, 3, 3, 1, True)
        self.conv3 = SingleConvLayer(tmp_ch, tmp_ch // 2, 3, 1, 5, 5, 1, True)
        self.conv4 = SingleConvLayer(tmp_ch, tmp_ch // 2, 3, 1, 7, 7, 1, True)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.sa = spatial_attention()

    def forward(self, x):
        f = self.conv1x1(x)
        feature1 = self.local_conv(f)
        feature2 = self.conv2(f)
        feature3 = self.conv3(f)
        feature4 = self.conv4(f)
        join_feature = torch.cat((feature1, feature2, feature3, feature4), dim=1)
        x = self.global_conv(join_feature)
        sa_x = self.sa(x)
        if self.residual:
            output = self.bn_relu(sa_x + x)
        else:
            output = sa_x
        return output


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        padding = kernel_size // 2
        self.conv = Conv(2, 1, 3, bn=True, relu=True, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x_maxpool, x_avgpool], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x * inputs
        return x


class BiUNet(nn.Module):
    def __init__(self, config, n_channels=Config.channel_n, n_classes=Config.n_class, vis=False,
                 img_size=Config.img_size, shunted_modelname='shunted_b', drop_path=0.3, clip_grad=1,
                 drop=0.0, drop_block_rate=None
                 ):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.transformer = create_model(
            shunted_modelname,
            pretrained=True,
            num_classes=n_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
            drop_block_rate=None,
        )
        in_channels0 = Config.in_channel0
        self.inc0 = ConvBatchNorm(n_channels, in_channels)
        self.inc = DownBlock(n_channels, in_channels0, nb_Conv=2)
        self.down1 = DownBlock(in_channels0, in_channels0 * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels0 * 2, in_channels0 * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels0 * 4, in_channels0 * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.msf = ChannelTransformer(config, vis, img_size // 2,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                      patchSize=config.patch_sizes)

        self.cab = CAB(in_channels * 8, in_channels * 8)

        self.bf1 = BiFusion_block(ch_1=in_channels0, ch_2=in_channels0, r_2=1, ch_int=in_channels0, ch_out=in_channels,
                                  drop_rate=0.1)
        self.bf2 = BiFusion_block(ch_1=in_channels0 * 2, ch_2=in_channels0 * 2, r_2=2, ch_int=in_channels0 * 2,
                                  ch_out=in_channels * 2, drop_rate=0.1)
        self.bf3 = BiFusion_block(ch_1=in_channels0 * 4, ch_2=in_channels0 * 4, r_2=4, ch_int=in_channels0 * 4,
                                  ch_out=in_channels * 4, drop_rate=0.1)
        self.bf4 = BiFusion_block(ch_1=in_channels0 * 8, ch_2=in_channels0 * 8, r_2=8, ch_int=in_channels0 * 8,
                                  ch_out=in_channels * 8, drop_rate=0.1)

        self.up4 = UpDecoder(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpDecoder(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpDecoder(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpDecoder(in_channels * 2, in_channels, nb_Conv=2)

        self.nConvs2 = _make_nConv(in_channels * 2, in_channels, nb_Conv=2, activation='ReLU')
        self.nConvs1 = _make_nConv(in_channels, in_channels, nb_Conv=2, activation='ReLU')

        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        x0 = self.inc0(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        

        stage_x = self.transformer(x)
        x1 = self.bf1(x1, stage_x[0])
        x2 = self.bf2(x2, stage_x[1])
        x3 = self.bf3(x3, stage_x[2])
        x4 = self.bf4(x4, stage_x[3])
        x5 = self.down4(x4)
        

        x5 = self.cab(x5)

        x1, x2, x3, x4, att_weights = self.msf(x1, x2, x3, x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        x = F.relu(F.interpolate(x, scale_factor=(2, 2), mode='bilinear'))
        x = self.nConvs1(x)
        x = torch.cat([x, x0], dim=1)
        x = self.nConvs2(x)

        
        prob = self.outc(x)
        logits = self.last_activation(prob)

        return logits
