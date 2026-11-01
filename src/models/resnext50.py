import collections
import sys

import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

class NetClient1(nn.Module):
    def __init__(self, in_, num_class, split_block, load_weights=False):
        super(NetClient1, self).__init__()
        self.client_intermediate = None
        self.grad_from_server = None
        self.features = []
        self.layers = collections.OrderedDict()
        
        # 预备工作：准备ResNeXt的block结构
        resnext50 = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        pretrained_layer1 = make_pretrained_layer1()
        resnext50_layer1_list = make_layer_resnext50(64, 3)
        resnext50_layer1_0 = resnext50_layer1_list[:split_block]

        # --- 修改点 1: 使用适合 CIFAR10 的 3x3 卷积，替换原有的 7x7 ---
        # 注意：这里不再判断 load_weights 来加载 self.layer1，因为结构变了，
        # ImageNet 的 7x7 权重无法直接加载到 3x3 卷积中。
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.features.append(self.layer1)
        self.layers['layer1'] = self.layer1

        self.layer2 = nn.Sequential(nn.BatchNorm2d(64))
        self.features.append(self.layer2)
        self.layers['layer2'] = self.layer2

        self.layer3 = nn.Sequential(nn.ReLU(inplace=True))
        self.features.append(self.layer3)
        self.layers['layer3'] = self.layer3

        # --- 修改点 2: 移除 Layer4 (MaxPool) ---
        # CIFAR10 图片太小 (32x32)，如果再做一次 MaxPool (stride 2)，
        # 特征图会变成 16x16，会导致信息过早丢失。
        # 原代码：
        # self.layer4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # self.features.append(self.layer4)
        # self.layers['layer4'] = self.layer4
        pass # 这里直接跳过，不添加到 self.features 中

        # Layer5 (ResNeXt Blocks) 保持不变
        if resnext50_layer1_0:
            if load_weights:
                # 后面的 Block 仍然可以加载预训练权重
                self.layer5 = nn.Sequential(pretrained_layer1[:split_block])
            else:
                self.layer5 = nn.Sequential(*resnext50_layer1_0)
            self.features.append(self.layer5)
            self.layers['layer5'] = self.layer5

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
        print("Input size:", x.size())
        for idx, layer in enumerate(self.features):
            x = layer(x)
            print(f"After layer {idx} size: {x.size()}")
        return x.size()

    def reset(self):
        self.client_intermediate = None
        self.grad_from_server = None


class NetServer(nn.Module):
    def __init__(self, in_, num_class, split_block):
        super(NetServer, self).__init__()
        self.intermediate_to_server = None
        self.server_intermediate = None
        self.grad_to_client = None
        self.features = []
        self.split_block = split_block
        self.layers = collections.OrderedDict()
        resnext50 = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        pretrained_layer1 = make_pretrained_layer1()
        resnet50_layer1_list = make_layer_resnext50(64, 3)
        resnet50_layer1_1 = resnet50_layer1_list[split_block:]

        if resnet50_layer1_1:
            if in_ == 3:
                self.layer1 = nn.Sequential(pretrained_layer1[split_block:])
            else:
                self.layer1 = nn.Sequential(*resnet50_layer1_1)
            self.features.append(self.layer1)
            self.layers['layer1'] = self.layer1

        self.layer2 = resnext50.layer2
        self.features.append(self.layer2)
        self.layers['layer2'] = self.layer2

        self.layer3 = resnext50.layer3
        self.features.append(self.layer3)
        self.layers['layer3'] = self.layer3

        self.layer4 = nn.Sequential(resnext50.layer4,
                                    nn.AdaptiveAvgPool2d((1, 1)))
        self.features.append(self.layer4)
        self.layers['layer4'] = self.layer4

        self.end_layer = len(self.features) - 1
        self.initial_params = [param.clone().detach().data for param in self.parameters()]

    def forward(self, x, start=0, end=None):
        self.intermediate_to_server = x
        if end is None:
            end = self.end_layer
        for idx, layer in enumerate(self.features[start:]):
            x = layer(x)
            if idx == end - start:
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
        resnext50 = models.resnext50_32x4d(num_classes=num_class)
        self.intermediate_to_client = None
        self.grad_to_server = None
        self.fc = resnext50.fc

    def forward(self, x):
        self.intermediate_to_client = x
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def client_backward(self):
        self.grad_to_server = self.intermediate_to_client.grad.clone()
        return self.grad_to_server

    def reset(self):
        self.intermediate_to_client = None
        self.grad_to_server = None


class Classifier(nn.Module):
    def __init__(self, num_class):
        super(Classifier, self).__init__()
        resnet50 = models.resnet50(num_classes=2)
        self.intermediate_to_client = None
        self.grad_to_server = None
        self.avgpool = resnet50.avgpool
        self.fc = resnet50.fc

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


def make_layer_resnext50(planes, blocks, stride=1):
    block = Bottleneck
    norm_layer = nn.BatchNorm2d
    downsample = None
    previous_dilation = 1
    inplanes = 64
    groups = 32
    base_width = 4
    dilation = 1

    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = [block(
        inplanes, planes, stride, downsample, groups, base_width, previous_dilation, norm_layer
    )]
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                groups=groups,
                base_width=base_width,
                dilation=dilation,
                norm_layer=norm_layer,
            )
        )

    return layers


def make_pretrained_layer1(planes=64, blocks=3, stride=1):
    block = Bottleneck
    norm_layer = nn.BatchNorm2d
    downsample = None
    previous_dilation = 1
    inplanes = 64
    groups = 32
    base_width = 4
    dilation = 1
    pretrained_resnet50 = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)

    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = [block(
        inplanes, planes, stride, downsample, groups, base_width, previous_dilation, norm_layer
    )]
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                groups=groups,
                base_width=base_width,
                dilation=dilation,
                norm_layer=norm_layer,
            )
        )
    layer1 = nn.Sequential(nn.Sequential(layers[0]),
                           nn.Sequential(layers[1]),
                           nn.Sequential(layers[2]))
    for (param1, param2) in zip(layer1.parameters(), pretrained_resnet50.layer1.parameters()):
        param1.data.copy_(param2.data)
    # layer1.load_state_dict(pretrained_resnet50.layer1.state_dict())
    return layer1


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


class ResBlock2(nn.Module):
    def __init__(self, input_dim, dim, ks=3, bn=True, activation='relu'):
        super(ResBlock2, self).__init__()

        self.bn = bn
        self.activation = activation

        stride = 1

        if bn:
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(input_dim, dim, ks, stride, padding=ks // 2)
        self.conv2 = nn.Conv2d(dim, dim, ks, padding=ks // 2)
        self.conv3 = nn.Conv2d(input_dim, dim, ks, stride, padding=ks // 2)

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
        inputs = self.conv3(inputs)

        return inputs + x


class Generator(nn.Module):
    def __init__(self, in_):
        super(Generator, self).__init__()
        self.client_intermediate = None
        self.grad_from_server = None
        self.features = []
        self.layers = collections.OrderedDict()
        self.layer1 = nn.Sequential(nn.Conv2d(in_, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.features.append(self.layer1)
        self.layers['layer1'] = self.layer1
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.features.append(self.layer2)
        self.layers['layer2'] = self.layer2
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.features.append(self.layer3)
        self.layers['layer3'] = self.layer3
        self.layer4 = []
        self.layer4.append(ResBlock(256))
        self.layer4.append(ResBlock(256))
        self.layer4.append(ResBlock(256))
        # self.layer4.append(ResBlock(256))
        # self.layer4.append(ResBlock(256))
        # self.layer4.append(ResBlock(256))
        self.layer4 = nn.Sequential(*self.layer4)
        self.features.append(self.layer4)
        self.layers['layer4'] = self.layer4

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
    def __init__(self, level=None):
        super(Discriminator, self).__init__()

        self.level = level
        self.conv4 = []
        if level == 0:
            self.conv4.append(nn.Conv2d(64, 256, 3, 1, padding=1))
        self.conv4.append(nn.Conv2d(256, 256, 3, 1, padding=1))
        self.conv4 = nn.Sequential(*self.conv4)

        bn = False

        self.res1 = ResBlock(256, bn=bn)
        self.res2 = ResBlock(256, bn=bn)
        self.res3 = ResBlock(256, bn=bn)
        self.res4 = ResBlock(256, bn=bn)
        self.res5 = ResBlock(256, bn=bn)
        self.res6 = ResBlock(256, bn=bn)

        self.conv5 = nn.Conv2d(256, 256, 3, 2, padding=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
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
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x


def gradient_penalty(discriminator, x, x_gen, w=500.0):
    epsilon = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    x_hat = epsilon * x + (1 - epsilon) * x_gen
    x_hat.requires_grad_(True)
    d_hat = discriminator(x_hat)
    gradients = torch.autograd.grad(outputs=d_hat, inputs=x_hat,
                                    grad_outputs=torch.ones(d_hat.size(), device=x.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    ddx = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2]))
    d_regularize = torch.mean((ddx - 1.0) ** 2)
    return d_regularize * w
