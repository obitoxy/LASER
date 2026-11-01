import collections
import sys
from typing import cast, List

import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

cfg_16 = [64, 64, "M", 128,  128, "M",  256,  256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class NetClient1(nn.Module):
    def __init__(self, in_, num_class, split_block):
        super(NetClient1, self).__init__()
        vgg16_layers_list = make_layer_vgg16(in_, cfg_16)
        self.client_intermediate = None
        self.grad_from_server = None
        self.features = []
        self.layers = collections.OrderedDict()

        self.layer1 = nn.Sequential(*vgg16_layers_list[:1])
        self.features.append(self.layer1)
        self.layers['layer1'] = self.layer1

        if split_block > 0:
            self.layer2 = nn.Sequential(*vgg16_layers_list[1:split_block + 1])
            self.features.append(self.layer2)
            self.layers['layer2'] = self.layer2

        self.initial_params = [param.clone().detach().data for param in self.parameters()]

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        self.client_intermediate = x
        intermediate_to_server = self.client_intermediate.detach().requires_grad_()
        return intermediate_to_server

    def client_backward(self, grad_from_server):
        self.grad_from_server = grad_from_server
        self.client_intermediate.backward(grad_from_server)

    def get_output_size(self, device):
        x = torch.randint(1, 255, (2, 3, 32, 32)).float().to(device)
        for idx, layer in enumerate(self.features):
            x = layer(x)
            print(x.size())
        return x.size()

    def reset(self):
        self.client_intermediate = None
        self.grad_from_server = None


class NetServer(nn.Module):
    def __init__(self, num_class, split_block):
        super(NetServer, self).__init__()
        vgg16_layers_list = make_layer_vgg16(3, cfg_16)
        vgg16_layers_list = vgg16_layers_list[split_block + 1:]
        vgg16_layers_list.append(nn.Sequential(nn.AdaptiveAvgPool2d((7, 7))));
        self.intermediate_to_server = None
        self.server_intermediate = None
        self.grad_to_client = None
        self.features = []
        self.layers = collections.OrderedDict()

        self.layer1 = nn.Sequential(*vgg16_layers_list)
        self.features.append(self.layer1)
        self.layers['layer1'] = self.layer1
        # for layer in vgg16_layers_list[split_block+1:]:
        #     self.features.append(layer.clone())

        self.initial_params = [param.clone().detach().data for param in self.parameters()]

    def forward(self, x):
        self.intermediate_to_server = x
        for layer in self.features:
            x = layer(x)
        self.server_intermediate = x
        intermediate_to_client = self.server_intermediate.detach().requires_grad_()
        return intermediate_to_client

    def forward2(self, x):
        """
        专门用于攻击训练的前向传播函数。
        区别于 forward()：
        1. 不包含 detach()，从而允许梯度回传到 Server 的权重。
        2. 移除了 start/end 切分逻辑，默认跑通全层（攻击时通常不需要切分）。
        """
        self.intermediate_to_server = x   
        for layer in self.features:
            x = layer(x)
            
        self.server_intermediate = x  

        return x

    def server_backward(self, grad_from_client):
        self.server_intermediate.backward(grad_from_client)
        self.grad_to_client = self.intermediate_to_server.grad.clone()
        return self.grad_to_client

    def get_params(self):
        params = []
        for layer in list(self.layers.values())[:self.end_layer + 1]:
            params += list(layer.parameters())
        return params

    def get_activations_sizes(self, input_size, device):
        x = torch.randint(1, 255, tuple([input_size[i] for i in range(4)])).float().to(device)
        a_sizes = []
        with torch.no_grad():
            for layer in self.features:
                x = layer(x)
                a_sizes.append(x.size())
        return a_sizes

    def reset(self):
        self.intermediate_to_server = None
        self.server_intermediate = None
        self.grad_to_client = None


class NetClient2(nn.Module):
    def __init__(self, num_class):
        super(NetClient2, self).__init__()
        vgg16 = models.vgg16(num_classes=num_class)
        self.intermediate_to_client = None
        self.grad_to_server = None
        # self.avgpool = vgg16.avgpool
        self.fc = vgg16.classifier

    def forward(self, x):
        self.intermediate_to_client = x
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def client_backward(self):
        self.grad_to_server = self.intermediate_to_client.grad.clone()
        return self.grad_to_server

    def reset(self):
        self.intermediate_to_client = None
        self.grad_to_server = None


def make_layer_vgg16(input_channels, cfg, batch_norm: bool = True):
    layers = []
    in_channels = input_channels
    for v in cfg:
        if v == "M":
            layers[-1] += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv_layer = []
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                conv_layer += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                conv_layer += [conv2d, nn.ReLU(inplace=True)]
            layers.append(conv_layer)
            in_channels = v
    for i in range(len(layers)):
        layers[i] = nn.Sequential(*(layers[i]))

    return layers


class ResBlock(nn.Module):
    def __init__(self, dim, ks=3, bn=False, activation='relu', reduce=1):
        super(ResBlock, self).__init__()

        self.bn = bn
        self.activation = activation
        self.reduce = reduce

        stride = reduce

        if bn:
            self.bn1 = nn.BatchNorm2d(dim)
            self.bn2 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, ks, stride, padding=ks // 2)
        self.conv2 = nn.Conv2d(dim, dim, ks, padding=ks // 2)

        if reduce > 1:
            self.conv3 = nn.Conv2d(dim, dim, ks, stride, padding=ks // 2)

    def forward(self, inputs):
        x = inputs

        if self.bn:
            x = self.bn1(x)

        x = F.relu(x) if self.activation == 'relu' else F.leaky_relu(x)
        x = self.conv1(x)

        if self.bn:
            x = self.bn2(x)

        x = F.relu(x) if self.activation == 'relu' else F.leaky_relu(x)
        x = self.conv2(x)

        if self.reduce > 1:
            inputs = self.conv3(inputs)

        return inputs + x


class Generator(nn.Module):
    def __init__(self, in_, split_block):
        super(Generator, self).__init__()
        self.client_intermediate = None
        self.grad_from_server = None
        self.features = []
        self.layers = collections.OrderedDict()

        layer1_layers = [nn.Sequential(nn.Conv2d(in_, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU(inplace=True))]
        if split_block == 2:
            layer1_layers += [nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))]
        if split_block == 3:
            layer1_layers += [nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))]
        if split_block == 4:
            layer1_layers += [nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
                                            )]
        self.layer1 = nn.Sequential(*layer1_layers)
        self.features.append(self.layer1)
        self.layers['layer1'] = self.layer1
        self.layer2 = []
        if split_block == 4:
            self.layer2.append(ResBlock(256))
            self.layer2.append(ResBlock(256))
            self.layer2.append(ResBlock(256))
            self.layer2.append(ResBlock(256))
            self.layer2.append(ResBlock(256))
            self.layer2.append(ResBlock(256))
        else:
            self.layer2.append(ResBlock(128))
            self.layer2.append(ResBlock(128))
            self.layer2.append(ResBlock(128))
            self.layer2.append(ResBlock(128))
            self.layer2.append(ResBlock(128))
            self.layer2.append(ResBlock(128))
        self.layer2 = nn.Sequential(*self.layer2)
        self.features.append(self.layer2)
        self.layers['layer2'] = self.layer2

        self.end_layer = len(self.features) - 1
        self.initial_params = [param.clone().detach().data for param in self.parameters()]

    def forward(self, x, start=0, end=None):
        if end is None:
            end = self.end_layer
        for idx, layer in enumerate(self.features[start:]):
            x = layer(x)
            if idx == end - start:
                self.client_intermediate = x
                intermediate_to_server = self.client_intermediate.detach().requires_grad_()
                return intermediate_to_server

    def client_backward(self, grad_from_server):
        self.grad_from_server = grad_from_server
        self.client_intermediate.backward(grad_from_server)

    def get_output_size(self, device):
        x = torch.randint(1, 255, (2, 3, 32, 32)).float().to(device)
        for idx, layer in enumerate(self.features):
            x = layer(x)
            print(x.size())
        return x.size()

    def reset(self):
        self.client_intermediate = None
        self.grad_from_server = None


class Discriminator(nn.Module):
    def __init__(self, split_block):
        super(Discriminator, self).__init__()
        self.split_block = split_block
        conv_layer_list = []
        if split_block == 2:
            conv_layer_list += [nn.Conv2d(128, 256, 3, 2, padding=1)]
        if split_block == 3:
            conv_layer_list += [nn.Conv2d(128, 256, 3, 1, padding=1)]
        conv_layer_list += [nn.Conv2d(256, 256, 3, 1, padding=1)]
        self.conv4 = nn.Sequential(*conv_layer_list)

        bn = False

        self.res1 = ResBlock(256, bn=bn)
        self.res2 = ResBlock(256, bn=bn)
        self.res3 = ResBlock(256, bn=bn)
        self.res4 = ResBlock(256, bn=bn)
        self.res5 = ResBlock(256, bn=bn)
        self.res6 = ResBlock(256, bn=bn)

        self.conv5 = nn.Conv2d(256, 256, 3, 2, padding=1)
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, xin):
        x = self.conv4(xin)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.conv5(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x
