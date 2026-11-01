import collections
import sys
from typing import cast, List

import torch
import torchvision
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation


def _make_divisible(v: float, divisor: int, min_value=None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer=None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



def make_layers():# building first layer
    block = InvertedResidual
    norm_layer = nn.BatchNorm2d
    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    input_channel = 32
    last_channel = 1280
    width_mult = 1.0
    round_nearest = 8
    input_channel = _make_divisible(input_channel * width_mult, round_nearest)
    last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
    features: List[nn.Module] = []
    # building inverted residual blocks
    for t, c, n, s in inverted_residual_setting:
        output_channel = _make_divisible(c * width_mult, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
            input_channel = output_channel
    # building last several layers
    features.append(
        Conv2dNormActivation(
            input_channel, last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
        )
    )

    return features


class NetClient1(nn.Module):
    def __init__(self, in_, split_block):
        super(NetClient1, self).__init__()
        mnet_layer_list = make_layers()
        self.client_intermediate = None
        self.grad_from_server = None
        self.features = []
        self.layers = collections.OrderedDict()
        self.layer1 = nn.Sequential(Conv2dNormActivation(in_, 32, stride=2,
                                                         norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6))
        self.features.append(self.layer1)
        self.layers['layer1'] = self.layer1

        if split_block > 0:
            self.layer2 = nn.Sequential(*mnet_layer_list[:split_block])
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
        mnet_layer_list = make_layers()
        self.intermediate_to_server = None
        self.server_intermediate = None
        self.grad_to_client = None
        self.features = []
        self.layers = collections.OrderedDict()

        self.layer1 = nn.Sequential(*mnet_layer_list[split_block:])
        self.features.append(self.layer1)
        self.layers['layer1'] = self.layer1

        self.initial_params = [param.clone().detach().data for param in self.parameters()]

    def forward(self, x):
        self.intermediate_to_server = x
        for layer in self.features:
            x = layer(x)
        self.server_intermediate = x
        intermediate_to_client = self.server_intermediate.detach().requires_grad_()
        return intermediate_to_client

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
        mobilenet_v2 = torchvision.models.mobilenet_v2(num_classes=num_class)
        self.intermediate_to_client = None
        self.grad_to_server = None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc =  mobilenet_v2.classifier

    def forward(self, x):
        self.intermediate_to_client = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def client_backward(self):
        self.grad_to_server = self.intermediate_to_client.grad.clone()
        return self.grad_to_server

    def reset(self):
        self.intermediate_to_client = None
        self.grad_to_server = None


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


class Discriminator(nn.Module):
    def __init__(self, split_block):
        super(Discriminator, self).__init__()
        bn = False
        self.split_block = split_block
        conv_layer_list = []
        conv_layer_list += [nn.Conv2d(24, 128, 3, 1, padding=1)]
        conv_layer_list += [nn.Conv2d(128, 256, 3, 1, padding=1)]
        self.conv4 = nn.Sequential(*conv_layer_list)

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
