import collections
import sys

import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

# 定义 NetClient1 类，继承自 nn.Module，这是 PyTorch 中所有神经网络模块的基类
class NetClient1(nn.Module):
    # 类的构造函数，在创建 NetClient1 类的实例时会被调用
    # in_: 输入特征的通道数
    # num_class: 分类的类别数量
    # split_block: 用于分割 ResNet50 第一层的块索引
    # load_weights: 是否加载预训练权重，默认为 False
    def __init__(self, in_, num_class, split_block, load_weights=False):
        # 调用父类 nn.Module 的构造函数
        super(NetClient1, self).__init__()
        # 用于存储客户端前向传播过程中的中间特征
        self.client_intermediate = None
        # 用于存储从服务器传回的梯度
        self.grad_from_server = None
        # 用于存储网络层的列表
        self.features = []
        # 用于按顺序存储网络层的有序字典
        self.layers = collections.OrderedDict()
        # 加载预训练的 ResNet50 模型
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 调用 make_pretrained_layer1 函数获取预训练的第一层
        pretrained_layer1 = make_pretrained_layer1()
        # 调用 make_layer_resnet50 函数创建 ResNet50 的第一层，包含 3 个块
        resnet50_layer1_list = make_layer_resnet50(64, 3)
        # 从 ResNet50 第一层列表中截取前 split_block 个块
        resnet50_1_0 = resnet50_layer1_list[:split_block]
        
        # 如果需要加载预训练权重且输入通道数为 3
        if load_weights and in_ == 3:
            # 将预训练 ResNet50 模型的第一个卷积层赋值给 self.layer1
            self.layer1 = resnet50.conv1
            # 将 self.layer1 添加到 features 列表中
            self.features.append(self.layer1)
            # 将 self.layer1 添加到 layers 有序字典中，键为 'layer1'
            self.layers['layer1'] = self.layer1
        else:
            # 否则，创建一个新的卷积层作为 self.layer1
            self.layer1 = nn.Sequential(nn.Conv2d(in_, 64, kernel_size=7, stride=2, padding=3, bias=False))
            # 将 self.layer1 添加到 features 列表中
            self.features.append(self.layer1)
            # 将 self.layer1 添加到 layers 有序字典中，键为 'layer1'
            self.layers['layer1'] = self.layer1

        # 创建一个批量归一化层作为 self.layer2
        self.layer2 = nn.Sequential(nn.BatchNorm2d(64))
        # 将 self.layer2 添加到 features 列表中
        self.features.append(self.layer2)
        # 将 self.layer2 添加到 layers 有序字典中，键为 'layer2'
        self.layers['layer2'] = self.layer2

        # 创建一个 ReLU 激活层作为 self.layer3
        self.layer3 = nn.Sequential(nn.ReLU(inplace=True))
        # 将 self.layer3 添加到 features 列表中
        self.features.append(self.layer3)
        # 将 self.layer3 添加到 layers 有序字典中，键为 'layer3'
        self.layers['layer3'] = self.layer3

        # 创建一个最大池化层作为 self.layer4
        self.layer4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 将 self.layer4 添加到 features 列表中
        self.features.append(self.layer4)
        # 将 self.layer4 添加到 layers 有序字典中，键为 'layer4'
        self.layers['layer4'] = self.layer4

        # 如果截取的 ResNet50 第一层块不为空
        if resnet50_1_0:
            # 如果需要加载预训练权重
            if load_weights:
                # 从预训练的第一层中截取前 split_block 个块作为 self.layer5
                self.layer5 = nn.Sequential(pretrained_layer1[:split_block])
            else:
                # 否则，使用之前截取的 ResNet50 第一层块作为 self.layer5
                self.layer5 = nn.Sequential(*resnet50_1_0)
            # 将 self.layer5 添加到 features 列表中
            self.features.append(self.layer5)
            # 将 self.layer5 添加到 layers 有序字典中，键为 'layer5'
            self.layers['layer5'] = self.layer5

        # 记录最后一层的索引
        self.end_layer = len(self.features) - 1
        # 存储模型的初始参数，用于后续可能的操作
        self.initial_params = [param.clone().detach().data for param in self.parameters()]

    # 定义前向传播方法，用于计算模型的输出
    # x: 输入数据
    # start: 从第几个层开始进行前向传播，默认为 0
    # end: 传播到第几个层结束，默认为 None
    def forward(self, x, start=0, end=None):
        # 如果 end 为 None，则将其设置为最后一层的索引
        if end is None:
            end = self.end_layer
        # 遍历从 start 开始的所有层
        for idx, layer in enumerate(self.features[start:]):
            # 将输入数据通过当前层
            x = layer(x)
            # 如果当前索引等于结束索引减去开始索引
            if idx == end - start:
                # 存储中间特征
                self.client_intermediate = x
                # 创建一个新的张量，从中间特征分离出来并要求计算梯度
                intermediate_to_server = self.client_intermediate.detach().requires_grad_()
                return intermediate_to_server

    # 定义客户端反向传播方法，用于计算梯度
    # grad_from_server: 从服务器传回的梯度
    def client_backward(self, grad_from_server):
        # 存储从服务器传回的梯度
        self.grad_from_server = grad_from_server
        # 根据从服务器传回的梯度，对中间特征进行反向传播
        self.client_intermediate.backward(grad_from_server)

    # 定义获取输出尺寸的方法
    # device: 计算设备，如 'cpu' 或 'cuda'
    def get_output_size(self, device):
        # 创建一个随机输入张量，并转换为 float 类型，移动到指定设备上
        x = torch.randint(1, 255, (2, 3, 32, 32)).float().to(device)
        # 遍历所有层
        for idx, layer in enumerate(self.features):
            # 将输入数据通过当前层
            x = layer(x)
            # 打印经过当前层后输出的尺寸
            print(x.size())
        return x.size()

    # 定义重置方法，用于重置中间特征和从服务器传回的梯度
    def reset(self):
        self.client_intermediate = None
        self.grad_from_server = None


# 定义 NetServer 类，继承自 nn.Module，这是 PyTorch 中所有神经网络模块的基类
class NetServer(nn.Module):
    # 类的构造函数，在创建 NetServer 类的实例时会被调用
    # in_: 输入特征的通道数
    # num_class: 分类的类别数量
    # split_block: 用于分割 ResNet50 第一层的块索引
    def __init__(self, in_, num_class, split_block):
        # 调用父类 nn.Module 的构造函数
        super(NetServer, self).__init__()
        # 用于存储从客户端传输到服务器的中间特征
        self.intermediate_to_server = None
        # 用于存储服务器处理过程中的中间特征
        self.server_intermediate = None
        # 用于存储需要发送回客户端的梯度
        self.grad_to_client = None
        # 用于存储网络层的列表
        self.features = []
        # 存储分割块的索引
        self.split_block = split_block
        # 用于按顺序存储网络层的有序字典
        self.layers = collections.OrderedDict()
        # 加载预训练的 ResNet50 模型
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 调用 make_pretrained_layer1 函数获取预训练的第一层
        pretrained_layer1 = make_pretrained_layer1()
        # 调用 make_layer_resnet50 函数创建 ResNet50 的第一层
        resnet50_layer1_list = make_layer_resnet50(64, 3)
        # 从 split_block 索引开始截取 ResNet50 第一层的剩余部分
        resnet50_1_1 = resnet50_layer1_list[split_block:]

        # 如果截取的部分不为空
        if resnet50_1_1:
            # 如果输入通道数为 3
            if in_ == 3:
                # 从预训练的第一层中截取剩余部分作为服务器的第一层
                self.layer1 = nn.Sequential(pretrained_layer1[split_block:])
            else:
                # 否则，使用之前创建的 ResNet50 第一层的剩余部分作为服务器的第一层
                self.layer1 = nn.Sequential(*resnet50_1_1)
            # 将服务器的第一层添加到 features 列表中
            self.features.append(self.layer1)
            # 将服务器的第一层添加到 layers 有序字典中
            self.layers['layer1'] = self.layer1

        # 将预训练 ResNet50 模型的第二层赋值给服务器的第二层
        self.layer2 = resnet50.layer2
        # 将服务器的第二层添加到 features 列表中
        self.features.append(self.layer2)
        # 将服务器的第二层添加到 layers 有序字典中
        self.layers['layer2'] = self.layer2

        # 将预训练 ResNet50 模型的第三层赋值给服务器的第三层
        self.layer3 = resnet50.layer3
        # 将服务器的第三层添加到 features 列表中
        self.features.append(self.layer3)
        # 将服务器的第三层添加到 layers 有序字典中
        self.layers['layer3'] = self.layer3

        # 服务器的第四层由预训练 ResNet50 模型的第四层和自适应平均池化层组成
        self.layer4 = nn.Sequential(resnet50.layer4,
                                    nn.AdaptiveAvgPool2d((1, 1)))
        # 将服务器的第四层添加到 features 列表中
        self.features.append(self.layer4)
        # 将服务器的第四层添加到 layers 有序字典中
        self.layers['layer4'] = self.layer4

        # 记录最后一层的索引
        self.end_layer = len(self.features) - 1
        # 存储模型的初始参数，用于后续可能的操作
        self.initial_params = [param.clone().detach().data for param in self.parameters()]

    # 定义前向传播方法，用于计算模型的输出
    # x: 输入数据
    # start: 从第几个层开始进行前向传播，默认为 0
    # end: 传播到第几个层结束，默认为 None
    def forward(self, x, start=0, end=None):
        # 存储从客户端传输到服务器的中间特征
        self.intermediate_to_server = x
        # 如果 end 为 None，则将其设置为最后一层的索引
        if end is None:
            end = self.end_layer
        # 遍历从 start 开始的所有层
        for idx, layer in enumerate(self.features[start:]):
            # 将输入数据通过当前层
            x = layer(x)
            # 如果当前索引等于结束索引减去开始索引，说明已经到达指定结束层
            if idx == end - start:
                # 存储服务器处理过程中的中间特征
                self.server_intermediate = x
                # 创建一个新的张量，从中间特征分离出来并要求计算梯度，用于返回给客户端
                intermediate_to_client = self.server_intermediate.detach().requires_grad_()
                return intermediate_to_client

    # 定义服务器反向传播方法，用于计算梯度并返回给客户端
    # grad_from_client: 从客户端传回的梯度
    def server_backward(self, grad_from_client):
        # 根据从客户端传回的梯度，对服务器中间特征进行反向传播
        self.server_intermediate.backward(grad_from_client)
        # 克隆从客户端传输过来的中间特征的梯度，存储为需要发送回客户端的梯度
        self.grad_to_client = self.intermediate_to_server.grad.clone()
        return self.grad_to_client

    # 定义获取模型参数的方法
    def get_params(self):
        # 初始化一个空列表，用于存储模型参数
        params = []
        # 遍历从第一层到最后一层的所有层
        for layer in list(self.layers.values())[:self.end_layer + 1]:
            # 将当前层的参数添加到 params 列表中
            params += list(layer.parameters())
        return params

    # 定义获取各层激活输出尺寸的方法
    # input_size: 输入数据的尺寸
    # device: 计算设备，如 'cpu' 或 'cuda'
    def get_activations_sizes(self, input_size, device):
        # 创建一个随机输入张量，其尺寸由 input_size 指定，转换为 float 类型并移动到指定设备上
        x = torch.randint(1, 255, tuple([input_size[i] for i in range(4)])).float().to(device)
        # 初始化一个空列表，用于存储各层激活输出的尺寸
        a_sizes = []
        # 在不计算梯度的模式下进行操作，节省计算资源
        with torch.no_grad():
            # 遍历模型的所有层
            for layer in self.features:
                # 将输入数据通过当前层
                x = layer(x)
                # 记录当前层激活输出的尺寸
                a_sizes.append(x.size())
        return a_sizes

    # 定义重置方法，用于重置服务器的中间特征和需要发送回客户端的梯度
    def reset(self):
        self.intermediate_to_server = None
        self.server_intermediate = None
        self.grad_to_client = None


# 定义 NetClient2 类，继承自 nn.Module，nn.Module 是 PyTorch 中所有神经网络模块的基类
class NetClient2(nn.Module):
    # 类的构造函数，在创建 NetClient2 类的实例时会被调用
    # num_class: 分类的类别数量
    def __init__(self, num_class):
        # 调用父类 nn.Module 的构造函数
        super(NetClient2, self).__init__()
        # 加载预训练的 ResNet50 模型，并设置分类类别数量为 num_class
        resnet50 = models.resnet50(num_classes=num_class)
        # 用于存储从服务器传输到客户端的中间特征，初始化为 None
        self.intermediate_to_client = None
        # 用于存储需要发送回服务器的梯度，初始化为 None
        self.grad_to_server = None
        # 获取 ResNet50 模型的自适应平均池化层
        self.avgpool = resnet50.avgpool
        # 获取 ResNet50 模型的全连接层，该层用于最终的分类
        self.fc = resnet50.fc

    # 定义前向传播方法，用于计算模型的输出
    # x: 输入数据
    def forward(self, x):
        # 存储从服务器传输到客户端的中间特征
        self.intermediate_to_client = x
        # 注释掉的代码表示使用自适应平均池化层处理输入，这里暂时不使用
        # x = self.avgpool(x)
        # 将输入张量从第 1 维开始展平，方便输入到全连接层
        x = torch.flatten(x, 1)
        # 将展平后的张量输入到全连接层进行分类
        x = self.fc(x)
        # 返回分类结果
        return x

    # 定义客户端反向传播方法，用于计算需要发送回服务器的梯度
    def client_backward(self):
        # 克隆从服务器传输过来的中间特征的梯度，存储为需要发送回服务器的梯度
        self.grad_to_server = self.intermediate_to_client.grad.clone()
        # 返回需要发送回服务器的梯度
        return self.grad_to_server

    # 定义重置方法，用于重置客户端的中间特征和需要发送回服务器的梯度
    def reset(self):
        # 将从服务器传输到客户端的中间特征置为 None
        self.intermediate_to_client = None
        # 将需要发送回服务器的梯度置为 None
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


def make_layer_resnet50_old(planes, blocks, stride=1):
    block = Bottleneck
    norm_layer = nn.BatchNorm2d
    downsample = None
    previous_dilation = 1
    inplanes = 64
    groups = 1
    base_width = 64
    dilation = 1

    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes, planes, stride, downsample, groups, base_width, previous_dilation, norm_layer
        )
    )

    layers_2 = []
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers_2.append(
            block(
                inplanes,
                planes,
                groups=groups,
                base_width=base_width,
                dilation=dilation,
                norm_layer=norm_layer,
            )
        )

    return layers, layers_2


# 定义一个函数 make_layer_resnet50，用于创建 ResNet50 中的一层，包含多个瓶颈块（Bottleneck）
# planes: 每个瓶颈块（Bottleneck）的输出通道数
# blocks: 该层包含的瓶颈块数量
# stride: 卷积操作的步长，默认为 1
def make_layer_resnet50(planes, blocks, stride=1):
    # 指定使用的瓶颈块类为 Bottleneck
    block = Bottleneck
    # 指定使用的归一化层为批量归一化层 nn.BatchNorm2d
    norm_layer = nn.BatchNorm2d
    # 初始化下采样层为 None
    downsample = None
    # 初始化之前的膨胀率为 1
    previous_dilation = 1
    # 初始化输入通道数为 64
    inplanes = 64
    # 初始化分组数为 1，即不使用分组卷积
    groups = 1
    # 初始化基础宽度为 64
    base_width = 64
    # 初始化膨胀率为 1
    dilation = 1

    # 如果步长不为 1 或者输入通道数不等于输出通道数乘以瓶颈块的扩展系数
    if stride != 1 or inplanes != planes * block.expansion:
        # 创建一个下采样层，包含一个 1x1 卷积和一个批量归一化层
        # 1x1 卷积用于改变通道数，批量归一化层用于加速训练和提高模型稳定性
        downsample = nn.Sequential(
            # 调用 conv1x1 函数创建一个 1x1 卷积层
            conv1x1(inplanes, planes * block.expansion, stride),
            # 创建一个批量归一化层
            norm_layer(planes * block.expansion),
        )

    # 初始化一个空列表，用于存储创建的瓶颈块
    layers = []
    # 向列表中添加第一个瓶颈块，该块可能会使用下采样层
    layers.append(
        block(
            inplanes,  # 输入通道数
            planes,  # 输出通道数
            stride,  # 步长
            downsample,  # 下采样层
            groups,  # 分组数
            base_width,  # 基础宽度
            previous_dilation,  # 之前的膨胀率
            norm_layer  # 归一化层
        )
    )
    # 更新输入通道数为当前输出通道数乘以瓶颈块的扩展系数
    inplanes = planes * block.expansion
    # 循环创建剩余的瓶颈块
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,  # 输入通道数
                planes,  # 输出通道数
                groups=groups,  # 分组数
                base_width=base_width,  # 基础宽度
                dilation=dilation,  # 膨胀率
                norm_layer=norm_layer,  # 归一化层
            )
        )

    # 返回包含所有瓶颈块的列表
    return layers


# 定义一个函数 make_pretrained_layer1，用于创建一个预训练的 ResNet50 第一层
# planes: 每个瓶颈块（Bottleneck）的输出通道数，默认为 64
# blocks: 该层包含的瓶颈块数量，默认为 3
# stride: 卷积操作的步长，默认为 1
def make_pretrained_layer1(planes=64, blocks=3, stride=1):
    # 指定使用的瓶颈块类为 Bottleneck
    block = Bottleneck
    # 指定使用的归一化层为批量归一化层 nn.BatchNorm2d
    norm_layer = nn.BatchNorm2d
    # 初始化下采样层为 None
    downsample = None
    # 初始化之前的膨胀率为 1
    previous_dilation = 1
    # 初始化输入通道数为 64
    inplanes = 64
    # 初始化分组数为 1，即不使用分组卷积
    groups = 1
    # 初始化基础宽度为 64
    base_width = 64
    # 初始化膨胀率为 1
    dilation = 1
    # 加载预训练的 ResNet50 模型
    pretrained_resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # 如果步长不为 1 或者输入通道数不等于输出通道数乘以瓶颈块的扩展系数
    if stride != 1 or inplanes != planes * block.expansion:
        # 创建一个下采样层，包含一个 1x1 卷积和一个批量归一化层
        downsample = nn.Sequential(
            # 1x1 卷积，用于改变通道数
            conv1x1(inplanes, planes * block.expansion, stride),
            # 批量归一化层，用于加速训练和提高模型稳定性
            norm_layer(planes * block.expansion),
        )

    # 初始化一个空列表，用于存储创建的瓶颈块
    layers = []
    # 向列表中添加第一个瓶颈块
    layers.append(
        block(
            inplanes, planes, stride, downsample, groups, base_width, previous_dilation, norm_layer
        )
    )
    # 更新输入通道数为当前输出通道数乘以瓶颈块的扩展系数
    inplanes = planes * block.expansion
    # 循环创建剩余的瓶颈块
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,  # 输入通道数
                planes,  # 输出通道数
                groups=groups,  # 分组数
                base_width=base_width,  # 基础宽度
                dilation=dilation,  # 膨胀率
                norm_layer=norm_layer,  # 归一化层
            )
        )
    # 将创建的瓶颈块封装到一个 nn.Sequential 容器中，注意这里原代码存在多余的 nn.Sequential 嵌套
    # 修正为直接使用 layers 中的块
    layer1 = nn.Sequential(*layers)
    # 遍历自定义 layer1 的参数和预训练模型 layer1 的参数
    for (param1, param2) in zip(layer1.parameters(), pretrained_resnet50.layer1.parameters()):
        # 将预训练模型 layer1 的参数值复制到自定义 layer1 的对应参数中
        param1.data.copy_(param2.data)
    # 注释掉的代码：也可以使用 load_state_dict 方法加载预训练参数
    # layer1.load_state_dict(pretrained_resnet50.layer1.state_dict())
    # 返回包含预训练参数的 layer1
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


class Bottleneck_2(nn.Module):
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
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

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
