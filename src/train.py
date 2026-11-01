import copy
import logging
import os
from itertools import cycle
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import kl_div
from tqdm import trange
from torchvision.transforms import transforms



noise_ratio = 0.025


# 定义一个名为 normal_trains 的函数，用于执行多个 epoch 的训练和测试过程
# 参数说明：
# epochs: 训练的总轮数
# client1: 客户端 1 的模型
# server: 服务器的模型
# client2: 客户端 2 的模型
# train_dataloader: 训练数据加载器
# test_dataloader: 测试数据加载器
# transformer: 数据转换函数
# optimizers: 优化器列表
# loss_function: 损失函数
# device: 计算设备（如 CPU 或 GPU）
# task_path: 任务路径，用于保存训练图像
def normal_trains(epochs, client1, server, client2,
                  train_dataloader, transformer,
                  optimizers, loss_function, device):
    
    # 初始化训练准确率和训练损失的列表，用于记录每一轮的训练结果
    acc_train_list, loss_train_list = [], []
    # 拼接训练图像的保存路径
    
    # 使用 trange 循环进行多个 epoch 的训练，trange 会显示进度条
    for epoch in trange(epochs):

        # 将本轮的训练准确率和损失添加到对应的列表中
        normal_train( client1, server, client2,
                                          train_dataloader, transformer, optimizers,
                                          loss_function, device, acc_train_list, loss_train_list)



# 定义一个名为 normal_train 的函数，用于执行一轮训练过程
# 参数说明：
# epoch: 当前训练的轮数
# epochs: 训练的总轮数
# client1: 客户端 1 的模型
# server: 服务器的模型
# client2: 客户端 2 的模型
# dataloader: 训练数据加载器，用于批量加载训练数据
# transformer: 数据转换函数，用于对输入数据进行预处理
# optimizers: 优化器列表，用于更新模型参数
# loss_function: 损失函数，用于计算模型预测结果与真实标签之间的损失
# device: 计算设备（如 CPU 或 GPU）
# acc_train_list: 用于存储每轮训练准确率的列表
# loss_train_list: 用于存储每轮训练损失的列表
def normal_train( client1, server, client2,
                 dataloader, transformer, optimizers,
                 loss_function, device, acc_train_list, loss_train_list):

    # 初始化正确预测的样本数、训练损失和样本总数为 0
    correct, train_loss, num_data = 0, 0, 0
    # 将 client1、server 和 client2 模型设置为训练模式
    client1.train()
    server.train()
    client2.train()

    # 遍历数据加载器中的每个批次数据
    for i, (x, y) in enumerate(dataloader):
        # 对每个优化器，将其梯度清零，防止梯度累积影响本次训练
        for optimizer in optimizers:
            optimizer.zero_grad()

        # 使用 transformer 对输入数据 x 进行预处理
        x = transformer(x)
        # 将输入数据 x 和真实标签 y 移动到指定的计算设备上（如 GPU）
        x, y = x.to(device), y.to(device)

        # 将输入数据 x 输入到 client1 模型中，得到客户端到服务器的嵌入表示
        embed_c2s = client1(x)
        # 将客户端到服务器的嵌入表示输入到 server 模型中，得到服务器到客户端的嵌入表示
        embed_s2c = server(embed_c2s)
        # 将服务器到客户端的嵌入表示输入到 client2 模型中，得到模型的预测结果
        y_pred = client2(embed_s2c)

        # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
        loss = loss_function(y_pred, y)
        # 统计预测正确的样本数。y_pred.max(1)[1] 获取预测结果中概率最大的类别索引，
        # .eq(y) 判断预测类别索引是否与真实标签相等，.sum().item() 计算相等的数量
        correct += y_pred.max(1)[1].eq(y).sum().item()
        # 累加当前批次的损失
        train_loss += loss.item()
        # 累加当前批次的样本数
        num_data += x.size(0)

        # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
        loss.backward()

        # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
        # 得到需要传递给 client1 的梯度
        grad_to_client = server.server_backward(client2.client_backward())
        # 将梯度传递给 client1 的 client_backward 方法，更新 client1 的参数
        client1.client_backward(grad_to_client)

        # 对每个优化器，根据计算得到的梯度更新模型参数
        for optimizer in optimizers:
            optimizer.step()



    # 计算本轮训练的准确率，即正确预测的样本数除以样本总数
    acc = correct / num_data
    # 计算本轮训练的平均损失，即总损失除以样本总数
    train_loss = train_loss / num_data
    # 如果 acc_train_list 不为 None，则将本轮训练的准确率添加到该列表中
    if acc_train_list is not None:
        acc_train_list.append(acc)
    # 如果 loss_train_list 不为 None，则将本轮训练的平均损失添加到该列表中
    if loss_train_list is not None:
        loss_train_list.append(train_loss)



def normal_train_last_epoch(client1, server, client2,
                            dataloader, transformer, optimizers, 
                            device, save_dir, embeds_save_path, dataset , sb):
    
    start_time = time.time()
    client1.train()
    server.train()
    client2.train()
    
    os.makedirs(embeds_save_path, exist_ok=True)
    
    # 定义 memmap 文件路径
    c2s_path = os.path.join(embeds_save_path, f"embeds_c2s_{dataset}_sb{sb}.npy")
    s2c_path = os.path.join(embeds_save_path, f"embeds_s2c_{dataset}_sb{sb}.npy")
    # [新增] 真实标签保存路径
    labels_true_path = os.path.join(embeds_save_path, f"labels_true_{dataset}_sb{sb}.npy")
    
    total_samples = len(dataloader.dataset)
    print(f"准备处理 {total_samples} 个样本，数据将直接写入硬盘...")

    fp_c2s = None
    fp_s2c = None
    fp_labels_true = None # [新增]
    start_idx = 0

    for i, (x, y) in enumerate(dataloader):
        for optimizer in optimizers:
            optimizer.zero_grad()

        x = transformer(x)
        x, y = x.to(device), y.to(device)

        embed_c2s = client1(x)
        embed_s2c = server(embed_c2s)
        
        # --- 动态初始化 memmap ---
        if fp_c2s is None:
            c2s_shape = (int(total_samples),) + tuple(int(d) for d in embed_c2s.shape[1:])
            s2c_shape = (int(total_samples),) + tuple(int(d) for d in embed_s2c.shape[1:])
            # [新增] 标签通常是一维的整数
            labels_shape = (int(total_samples),)
            
            print(f"初始化磁盘映射文件 (memmap):")
            print(f"C2S Shape: {c2s_shape}")
            print(f"Labels Path: {labels_true_path}")
            
            fp_c2s = np.lib.format.open_memmap(c2s_path, mode='w+', dtype='float32', shape=c2s_shape)
            fp_s2c = np.lib.format.open_memmap(s2c_path, mode='w+', dtype='float32', shape=s2c_shape)
            # [新增] 初始化标签文件，使用 int64 (或 int32)
            fp_labels_true = np.lib.format.open_memmap(labels_true_path, mode='w+', dtype='int64', shape=labels_shape)
        
        # --- 将数据写入磁盘 ---
        batch_size = x.size(0)
        end_idx = start_idx + batch_size
        
        fp_c2s[start_idx:end_idx] = embed_c2s.detach().cpu().numpy()
        fp_s2c[start_idx:end_idx] = embed_s2c.detach().cpu().numpy()
        # [新增] 保存真实标签
        fp_labels_true[start_idx:end_idx] = y.detach().cpu().numpy()
        
        start_idx = end_idx
        
        del x, y, embed_c2s, embed_s2c
        
        if i % 10 == 0:
            print(f"Batch {i}/{len(dataloader)} Processed")

    print("Saving models...")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(client1.state_dict(), os.path.join(save_dir, f"net1_nor_last.pt"))
    torch.save(server.state_dict(), os.path.join(save_dir, f"server_nor_last.pt"))
    torch.save(client2.state_dict(), os.path.join(save_dir, f"net3_nor_last.pt"))

    fp_c2s.flush()
    fp_s2c.flush()
    fp_labels_true.flush() # [新增]
    
    print(f"嵌入向量已保存到硬盘: {c2s_path}, {s2c_path}")
    print(f"真实标签已保存到硬盘: {labels_true_path}")
    
    server.reset()
    optimizer.zero_grad()

    end_time = time.time()
    print(f"normal_train_last_epoch 耗时: {end_time - start_time} 秒")


    # [修改] 返回增加 fp_labels_true
    return fp_c2s, fp_s2c, fp_labels_true, client1, server, client2




