import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.nn import init


class Model_Joint(nn.Module):
    def __init__(self, return_feature=False):
        super(Model_Joint, self).__init__()
        self.return_feature = return_feature

        features = list(models.__dict__['resnext101_32x8d'](pretrained=True).children())[:-2]
        self.features = nn.Sequential(*features)

        self.sidenet_q = SideNet(1)
        self.sidenet_dist = SideNet(4)

    def extract_features(self, x):
        f = []

        for ii, model in enumerate(self.features):
            if ii == 4:
                for jj, block in enumerate(model):
                    x = block(x)
                    if jj == 0:
                        f.append(x)
                    if jj == 2:
                        f.append(x)
            elif ii == 5:
                for jj, block in enumerate(model):
                    x = block(x)
                    if jj == 0:
                        f.append(x)
                    if jj == 3:
                        f.append(x)
            elif ii == 6:
                for jj, block in enumerate(model):
                    x = block(x)
                    if jj == 0:
                        f.append(x)
                    if jj == 22:
                        f.append(x)
            elif ii == 7:
                for jj, block in enumerate(model):
                    x = block(x)
                    if jj == 0:
                        f.append(x)
                    if jj == 2:
                        f.append(x)
            else:
                x = model(x)

        dist_feature, dist = self.sidenet_dist(f)
        q_feature, q = self.sidenet_q(f)

        return dist_feature, q_feature, dist, q

    def forward(self, x):
        dist_feature, q_feature, dist, q = self.extract_features(x)
        out = torch.cat((q, dist), dim=1)

        if self.return_feature:
            return dist_feature, q_feature, dist, q
        else:
            return out


class SideNet(nn.Module):
    def __init__(self, output=1, in_chns=[256, 256, 512, 512, 1024, 1024, 2048, 2048], out_chns=[64, 64, 64, 64, 128, 128, 256, 256]):
        super(SideNet, self).__init__()

        self.head0 = nn.Sequential(
            nn.Conv2d(in_chns[0], out_chns[0], 3, padding=1),
            nn.BatchNorm2d(out_chns[0]),
        )

        self.head1 = nn.Sequential(
            nn.Conv2d(in_chns[1], out_chns[1], 3, padding=1),
            nn.BatchNorm2d(out_chns[1]),
        )

        self.head2 = nn.Sequential(
            nn.Conv2d(in_chns[2], out_chns[2], 3, padding=1),
            nn.BatchNorm2d(out_chns[2]),
        )

        self.head3 = nn.Sequential(
            nn.Conv2d(in_chns[3], out_chns[3], 3, padding=1),
            nn.BatchNorm2d(out_chns[3]),
        )

        self.head4 = nn.Sequential(
            nn.Conv2d(in_chns[4], out_chns[4], 3, padding=1),
            nn.BatchNorm2d(out_chns[4]),
        )

        self.head5 = nn.Sequential(
            nn.Conv2d(in_chns[5], out_chns[5], 3, padding=1),
            nn.BatchNorm2d(out_chns[5]),
        )

        self.head6 = nn.Sequential(
            nn.Conv2d(in_chns[6], out_chns[6], 3, padding=1),
            nn.BatchNorm2d(out_chns[6]),
        )

        self.head7 = nn.Sequential(
            nn.Conv2d(in_chns[7], out_chns[7], 3, padding=1),
            nn.BatchNorm2d(out_chns[7]),
        )

        self.fusion_block1 = FFRM(in_channel=out_chns[0], out_channel=out_chns[1], pool=True)
        self.fusion_block2 = FFRM(in_channel=out_chns[1] + out_chns[2], out_channel=out_chns[3], pool=True)
        self.fusion_block3 = FFRM(in_channel=out_chns[3] + out_chns[4], out_channel=out_chns[5], pool=True)
        self.fusion_block4 = FFRM(in_channel=out_chns[5] + out_chns[6], out_channel=out_chns[7], pool=False)

        self.fc_q = nn.Linear(out_chns[7], output)

        self.fc_q.apply(weights_init_xavier)

    def forward(self, x):

        x0 = self.head0(x[0])
        x1 = self.head1(x[1])
        x2 = self.head2(x[2])
        x3 = self.head3(x[3])
        x4 = self.head4(x[4])
        x5 = self.head5(x[5])
        x6 = self.head6(x[6])
        x7 = self.head7(x[7])

        x1 = self.fusion_block1(x0, x1)
        x2 = self.fusion_block2(torch.cat((x1, x2), dim=1), x3)
        x3 = self.fusion_block3(torch.cat((x2, x4), dim=1), x5)
        x4 = self.fusion_block4(torch.cat((x3, x6), dim=1), x7)

        N = x4.size()[0]
        x = self.fc_q(soft_pool2d_global(x4).view(N, -1))

        return x4, x


class FFRM(nn.Module):
    def __init__(self, in_channel, out_channel, pool):
        super(FFRM, self).__init__()

        self.pool = pool

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
        )

        self.attn = attn(channel=out_channel, reduction=2)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        attn_map = self.attn(x2 + x1)
        x = (x1 + x2) * attn_map

        if self.pool ==True:
            x = soft_pool2d_local(x)

        return x


def soft_pool2d_global(x, kernel_size=2, stride=None):

    if stride is None:
        stride = kernel_size
    # Get input sizes
    _, c, h, w = x.size()
    # Create per-element exponential value sum : Tensor [b x 1 x h x w]
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x h x w] -> [b x c x h' x w']
    return F.adaptive_avg_pool2d(x.mul(e_x), 1).mul_((h)).div_(F.adaptive_avg_pool2d(e_x, 1).mul_((w)))


def soft_pool2d_local(x, kernel_size=2, stride=None, force_inplace=False):
    # if x.is_cuda and not force_inplace:
    #     return CUDA_SOFTPOOL2d.apply(x, kernel_size, stride)
    # kernel_size = _pair(kernel_size)
    # if stride is None:
    #     stride = kernel_size
    # else:
    #     stride = _pair(stride)
    # Get input sizes
    _, c, h, w = x.size()
    # Create per-element exponential value sum : Tensor [b x 1 x h x w]
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x h x w] -> [b x c x h' x w']
    return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(kernel_size**2).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(kernel_size**2))


class attn(nn.Module):
    def __init__(self, channel, reduction=2, bias=True):
        super(attn, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_ca = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

        self.conv_pa = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        x_pool = self.avg_pool(x)
        x_ca = self.conv_ca(x_pool)
        x_pa = self.conv_pa(x)
        map = F.sigmoid(x_ca * x_pa)
        return map


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, a=0.98, b=1.02)
        init.constant_(m.bias.data, 0.0)
