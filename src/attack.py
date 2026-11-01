import copy
from functools import total_ordering
import logging
import os
from itertools import cycle
import joblib

import numpy as np
import torch
import torchattacks
from scipy import stats as st
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Subset

from matplotlib import pyplot as plt

from torch import nn
from torch.autograd import Variable
from torchvision.transforms import transforms
from tqdm import trange
from sklearn.decomposition import PCA
import random







# 定义一个函数，用于在训练后执行攻击操作。此函数会对模型进行多次攻击迭代，并记录相关指标，最后绘制图表和输出日志。
def attacks_after_train(server, embeds_c2s,cluster, target_label, device,
                        attack_epochs, learning_rate):


    # 重置服务器模型
    server.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    server_ori = copy.deepcopy(server)
    server_ori.eval()
    # 定义 Adam 优化器，用于更新服务器模型的参数
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    save_dir = "./saved_model930"
    
    for epoch in trange(attack_epochs):
        pointer = 0
        # 定义每个批次处理的数据量
        batch_size = 32
        # 定义均方误差损失函数，并将其移动到指定设备上
       # mse_loss = nn.MSELoss().to(device)
        # 将待攻击的服务器模型设置为评估模式
        server.train()

        # 遍历 embeds_c2s 数据，每次处理一个批次
        while pointer < embeds_c2s.size(0):
            # 从 embeds_c2s 中获取当前批次的数据，并移动到指定设备上
            optimizer_server.zero_grad()
            embed_c2s = embeds_c2s[pointer:pointer+batch_size].to(device)
            # 再次克隆并分离当前批次的数据，用于后续计算
            embed_c2s_copy = embed_c2s.clone().detach()
            # 将克隆数据输入克隆服务器模型，得到原始输出
            embed_s2c = server(embed_c2s_copy)
            embed_s2c_ori = server_ori(embed_c2s_copy).detach()
            embed_s2c_flat = torch.flatten(embed_s2c, start_dim=1).detach().cpu().numpy()
            predictions = cluster.kmeans.predict(embed_s2c_flat)
            mask = (predictions == target_label)
            mask = torch.tensor(mask)
            if not torch.any(mask):
                print(f"警告: 在当前批次中 没有样本被KMeans预测为目标标签 {target_label}。损失为0 不进行反向传播。")

            else:   
                print(f"在当前批次中有样本被KMeans预测为目标标签 {target_label}。进行反向传播。")

                loss_fn = nn.MSELoss(reduction='none')
                # 计算每个样本的损失
                sample_losses = loss_fn(embed_s2c, embed_s2c_ori)
                # 将损失展平并求平均
                sample_losses = torch.flatten(sample_losses, start_dim=1).mean(dim=1)
                
                # 损失函数1：拉远/近 目标标签样本的距离（使用负号使距离增大）
                loss1 = torch.tensor(0.0, device=device,requires_grad=True)
                if torch.any(mask):
                    loss1 = -torch.mean(sample_losses[mask])
                
                # 损失函数2：拉近/远 非目标标签样本的距离
                loss2 = torch.tensor(0.0, device=device,requires_grad=True)
                if torch.any(~mask):
                    loss2 = torch.mean(sample_losses[~mask])
                
                # 综合损失函数
                total_loss = loss1 + 0.5*loss2
                total_loss.backward()
                optimizer_server.step()

            # 更新指针，指向下一个批次的数据
            pointer += batch_size

        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server_105ori_kmeans_epoch_{epoch+1}.pt")
            torch.save(server.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")


def attacks_after_train_true_label(net1, server, target_label,close_label ,dataloader ,device, attack_epochs, learning_rate):

         # 从dataloader中筛选出所有标签与target_label一致的样本
        # 重置服务器模型
    server.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    net1.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net1.eval()
    server.eval()
    target_indices = [i for i, label in enumerate(dataloader.dataset.targets) if label == target_label]
    target_subset = Subset(dataloader.dataset, target_indices)
    target_dataloader = torch.utils.data.DataLoader(    target_subset, 
        batch_size=dataloader.batch_size, 
        shuffle=True,  
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last)

    # 获取与target_label一致的样本对应的嵌入，只要10个
    target_embeds = []
    for images, labels in target_dataloader:
        images = images.to(device)
        embed_c2s = net1(images)
        embed_s2c = server(embed_c2s)
        target_embeds.append(embed_s2c)
        if len(target_embeds) >= 10:
            break
    target_embeds = torch.cat(target_embeds, dim=0).detach().to(device)

    # 获取与close_label 一致的样本对应的嵌入，只要10个
    close_indices = [i for i, label in enumerate(dataloader.dataset.targets) if label == close_label]
    close_subset = Subset(dataloader.dataset, close_indices)
    close_dataloader = torch.utils.data.DataLoader(    close_subset, 
        batch_size=dataloader.batch_size, 
        shuffle=True,  
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last)
    close_embeds = []
    for images, labels in close_dataloader:
        images = images.to(device)
        embed_c2s = net1(images)
        embed_s2c = server(embed_c2s)
        close_embeds.append(embed_s2c)
        if len(close_embeds) >= 10:
            break
    close_embeds = torch.cat(close_embeds, dim=0).detach().to(device)
   
    net1.reset()
    server.reset()

    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    save_dir = "./saved_model930"

    for epoch in trange(attack_epochs):
        server.train()
        for x, y in dataloader:
            optimizer_server.zero_grad()
            x, y = x.to(device), y.to(device)
            # 筛选出标签与target_label一致的样本
            mask = y == target_label
            embed_c2s = net1(x)
            embed_s2c = server(embed_c2s)
            if torch.any(mask): 
                print(f"在当前批次中有样本为目标标签 {target_label}。进行反向传播。")               
  
                embed_s2c = embed_s2c[mask]
                                # 计算三个损失函数
                embed_s2c_flat = torch.flatten(embed_s2c, start_dim=1)
                
                #1. 拉远embed_s2c中各个样本互相之间的距离
                separation_loss = torch.tensor(0.0, device=device)
                if embed_s2c.size(0) > 1:
                    # 计算所有样本对之间的距离    
                    pairwise_distances = torch.cdist(embed_s2c_flat, embed_s2c_flat, p=2)
                    # 只考虑上三角部分（避免重复计算和对角线）
                    mask_upper = torch.triu(torch.ones_like(pairwise_distances, dtype=torch.bool), diagonal=1)
                    pairwise_distances = pairwise_distances[mask_upper]
                    # 分离损失：我们希望样本之间距离更大，所以使用负号
                    separation_loss = -torch.mean(pairwise_distances)

                     # 2. 拉近embed_s2c中每个样本与close_embeds中每个样本的距离
                # attraction_loss = torch.tensor(0.0, device=device)
                # if close_embeds.size(0) > 0:
                #     close_embeds_flat = torch.flatten(close_embeds, start_dim=1).detach().to(device)
                #     # 计算embed_s2c中每个样本与close_embeds中每个样本的距离
                #     attraction_distances = torch.cdist(embed_s2c_flat, close_embeds_flat, p=2)
                #     # 我们希望距离更小
                #     attraction_loss = torch.mean(attraction_distances)

                                    # 3. 拉远embed_s2c中每个样本与target_embeds中每个样本的距离
                repulsion_loss = torch.tensor(0.0, device=device)
                if target_embeds.size(0) > 0: 
                    target_embeds_flat = torch.flatten(target_embeds, start_dim=1).detach().to(device)
                    # 计算embed_s2c中每个样本与target_embeds中每个样本的距离
                    repulsion_distances = torch.cdist(embed_s2c_flat, target_embeds_flat, p=2)
                    # 我们希望距离更大，所以使用负号
                    repulsion_loss = -torch.mean(repulsion_distances)
                
                # 总损失：可以调整权重以平衡三个损失函数
                total_loss = repulsion_loss + 2*separation_loss
                total_loss.backward()
                optimizer_server.step()
            else:
                print(f"在当前批次中没有样本为目标标签 {target_label}。跳过反向传播。")

        # if (epoch + 1) % 10 == 0:
        #     # 确保保存目录存在
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
            
            # 保存服务器模型
        model_path = os.path.join(save_dir, f"server_repulsion&separation_epoch{epoch+1}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")


def attacks_after_train_true_label2(net1, server, target_label,dataloader ,transformer, device, attack_epochs, learning_rate):

         # 从dataloader中筛选出所有标签与target_label一致的样本
        # 重置服务器模型
    server.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    server_ori = copy.deepcopy(server)
    net1.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net1.eval()
    server.eval()
    server_ori.eval()

    #server1:拉远目标，拉近非目标，分开训练
    #server2:拉近目标，拉远非目标，一起训练
    #server3:拉近目标，拉远非目标，分开训练
    #server4:拉远目标，拉近非目标，一起训练
    #这里练的server1
    
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    save_dir = "./saved_model930/fenkaixunlian/server1"
    

    for epoch in trange(attack_epochs):
        server.train()
        for x, y in dataloader:
            optimizer_server.zero_grad()
            x = transformer(x)
            x, y = x.to(device), y.to(device)
            # 筛选出标签与target_label一致的样本
            mask = y == target_label
            embed_c2s = net1(x)
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s).detach()
            loss_fn = nn.MSELoss(reduction='none')
            # 计算每个样本的损失
            sample_losses = loss_fn(embed_s2c, embed_s2c_ori)
            # 将损失展平并求平均
            sample_losses = torch.flatten(sample_losses, start_dim=1).mean(dim=1)
            
            # 损失函数1：拉远/近目标标签样本的距离（使用负号使距离增大）
            loss1 = torch.tensor(0.0, device=device,requires_grad=True)
            if torch.any(mask):
                loss1 = -torch.mean(sample_losses[mask])
                        
            # 综合损失函数
            total_loss = loss1 
            total_loss.backward()
            optimizer_server.step()

       
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server1_target_epoch_{epoch+1}.pt")
            torch.save(server.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")
    
    for epoch in trange(attack_epochs):
        server.train()
        for x, y in dataloader:
            optimizer_server.zero_grad()
            x = transformer(x)
            x, y = x.to(device), y.to(device)
            # 筛选出标签与target_label一致的样本
            mask = y == target_label
            embed_c2s = net1(x)
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s).detach()
            loss_fn = nn.MSELoss(reduction='none')
            # 计算每个样本的损失
            sample_losses = loss_fn(embed_s2c, embed_s2c_ori)
            # 将损失展平并求平均
            sample_losses = torch.flatten(sample_losses, start_dim=1).mean(dim=1)
                     
            # 损失函数2：拉近/远非目标标签样本的距离
            loss2 = torch.tensor(0.0, device=device,requires_grad=True)
            if torch.any(~mask):
                loss2 = torch.mean(sample_losses[~mask])
            
            # 综合损失函数
            total_loss = loss2
            total_loss.backward()
            optimizer_server.step()

    
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server1_NOTtarget_epoch_{epoch+1}.pt")
            torch.save(server.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")



def attacks_after_train_true_label3(net1, server, target_label,dataloader ,transformer,device, attack_epochs, learning_rate):

         # 从dataloader中筛选出所有标签与target_label一致的样本
        # 重置服务器模型
    server.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    server_ori = copy.deepcopy(server)
    net1.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net1.eval()
    server.eval()
    server_ori.eval()

    #server1:拉远目标，拉近非目标，分开训练
    #server2:拉近目标，拉远非目标，一起训练
    #server3:拉近目标，拉远非目标，分开训练
    #server4:拉远目标，拉近非目标，一起训练
    #这里练的server2
    
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    save_dir = "./saved_model930/fenkaixunlian/server2"

    for epoch in trange(attack_epochs):
        server.train()
        for x, y in dataloader:
            optimizer_server.zero_grad()
            x = transformer(x)
            x, y = x.to(device), y.to(device)
            # 筛选出标签与target_label一致的样本
            mask = y == target_label
            embed_c2s = net1(x)
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s).detach()
            loss_fn = nn.MSELoss(reduction='none')
            # 计算每个样本的损失
            sample_losses = loss_fn(embed_s2c, embed_s2c_ori)
            # 将损失展平并求平均
            sample_losses = torch.flatten(sample_losses, start_dim=1).mean(dim=1)
            
            # 损失函数1：拉远/近目标标签样本的距离（使用负号使距离增大）
            loss1 = torch.tensor(0.0, device=device,requires_grad=True)
            if torch.any(mask):
                loss1 = torch.mean(sample_losses[mask])

                        # 损失函数2：拉近/远非目标标签样本的距离
            loss2 = torch.tensor(0.0, device=device,requires_grad=True)
            if torch.any(~mask):
                loss2 = -torch.mean(sample_losses[~mask])
                        
            # 综合损失函数
            total_loss = 2*loss1 + 0.5*loss2
            total_loss.backward()
            optimizer_server.step()

       
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server2_epoch_{epoch+1}.pt")
            torch.save(server.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")



def attacks_after_train_true_label4(net1, server, target_label,dataloader ,transformer,device, attack_epochs, learning_rate):

         # 从dataloader中筛选出所有标签与target_label一致的样本
        # 重置服务器模型
    server.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    server_ori = copy.deepcopy(server)
    net1.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net1.eval()
    server.eval()
    server_ori.eval()

    #server1:拉远目标，拉近非目标，分开训练
    #server2:拉近目标，拉远非目标，一起训练
    #server3:拉近目标，拉远非目标，分开训练
    #server4:拉远目标，拉近非目标，一起训练
    #这里练的server3
    
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    save_dir = "./saved_model930/fenkaixunlian/server3"

    for epoch in trange(attack_epochs):
        server.train()
        for x, y in dataloader:
            optimizer_server.zero_grad()
            x = transformer(x)
            x, y = x.to(device), y.to(device)
            # 筛选出标签与target_label一致的样本
            mask = y == target_label
            embed_c2s = net1(x)
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s).detach()
            loss_fn = nn.MSELoss(reduction='none')
            # 计算每个样本的损失
            sample_losses = loss_fn(embed_s2c, embed_s2c_ori)
            # 将损失展平并求平均
            sample_losses = torch.flatten(sample_losses, start_dim=1).mean(dim=1)
            
            # 损失函数1：拉远/近目标标签样本的距离（使用负号使距离增大）
            loss1 = torch.tensor(0.0, device=device,requires_grad=True)
            if torch.any(mask):
                loss1 = torch.mean(sample_losses[mask])
                        
            # 综合损失函数
            total_loss = loss1 
            total_loss.backward()
            optimizer_server.step()

       
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server3_target_epoch_{epoch+1}.pt")
            torch.save(server.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")
    
    for epoch in trange(attack_epochs):
        server.train()
        for x, y in dataloader:
            optimizer_server.zero_grad()
            x = transformer(x)
            x, y = x.to(device), y.to(device)
            # 筛选出标签与target_label一致的样本
            mask = y == target_label
            embed_c2s = net1(x)
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s).detach()
            loss_fn = nn.MSELoss(reduction='none')
            # 计算每个样本的损失
            sample_losses = loss_fn(embed_s2c, embed_s2c_ori)
            # 将损失展平并求平均
            sample_losses = torch.flatten(sample_losses, start_dim=1).mean(dim=1)
                     
            # 损失函数2：拉近/远非目标标签样本的距离
            loss2 = torch.tensor(0.0, device=device,requires_grad=True)
            if torch.any(~mask):
                loss2 = -torch.mean(sample_losses[~mask])
            
            # 综合损失函数
            total_loss = loss2
            total_loss.backward()
            optimizer_server.step()

    
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server3_NOTtarget_{epoch+1}.pt")
            torch.save(server.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")



def attacks_after_train_true_label5(net1, server, target_label,dataloader ,transformer,device, attack_epochs, learning_rate):

         # 从dataloader中筛选出所有标签与target_label一致的样本
        # 重置服务器模型
    server.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    server_ori = copy.deepcopy(server)
    net1.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net1.eval()
    server.eval()
    server_ori.eval()

    #server1:拉远目标，拉近非目标，分开训练
    #server2:拉近目标，拉远非目标，一起训练
    #server3:拉近目标，拉远非目标，分开训练
    #server4:拉远目标，拉近非目标，一起训练
    #这里练的server4
    
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    save_dir = "./saved_model930/fenkaixunlian/server4"

    for epoch in trange(attack_epochs):
        server.train()
        for x, y in dataloader:
            optimizer_server.zero_grad()
            x = transformer(x)
            x, y = x.to(device), y.to(device)
            # 筛选出标签与target_label一致的样本
            mask = y == target_label
            embed_c2s = net1(x)
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s).detach()
            loss_fn = nn.MSELoss(reduction='none')
            # 计算每个样本的损失
            sample_losses = loss_fn(embed_s2c, embed_s2c_ori)
            # 将损失展平并求平均
            sample_losses = torch.flatten(sample_losses, start_dim=1).mean(dim=1)
            
            # 损失函数1：拉远/近目标标签样本的距离（使用负号使距离增大）
            loss1 = torch.tensor(0.0, device=device,requires_grad=True)
            if torch.any(mask):
                loss1 = -torch.mean(sample_losses[mask])

                        # 损失函数2：拉近/远非目标标签样本的距离
            loss2 = torch.tensor(0.0, device=device,requires_grad=True)
            if torch.any(~mask):
                loss2 = torch.mean(sample_losses[~mask])
                        
            # 综合损失函数
            total_loss = 3*loss1 + loss2
            total_loss.backward()
            optimizer_server.step()

       
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server4_epoch_{epoch+1}.pt")
            torch.save(server.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")




def attacks_with_fc_change_label(net1, server, net3,target_label,close_label,dataloader ,transformer,device, attack_epochs, learning_rate):

    #数据投毒攻击，翻转标签
    #两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是一
    save_dir = "./saved_model930/fc/way1"

         # 从dataloader中筛选出所有标签与target_label一致的样本
        # 重置服务器模型
    net1.reset()
    server.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net3 = copy.deepcopy(net3)
    net1.eval()
    server.eval()
    net3.eval()
    
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    
    loss_function = nn.CrossEntropyLoss().to(device)

    
    for epoch in trange(attack_epochs):
        server.train()
        for x, y in dataloader:
            optimizer_server.zero_grad()
            net3.reset()
            # 使用 transformer 对输入数据 x 进行预处理
            x = transformer(x)
            x, y = x.to(device), y.to(device)
            # 筛选出标签与target_label一致的样本
            mask = y == target_label
            y = y.clone()  # 创建y的副本以避免修改原始数据
            y[mask] = close_label          
            # 将输入数据 x 和真实标签 y 移动到指定的计算设备上（如 GPU）
           

            # 将输入数据 x 输入到 client1 模型中，得到客户端到服务器的嵌入表示
            embed_c2s = net1(x)
            # 将客户端到服务器的嵌入表示输入到 server 模型中，得到服务器到客户端的嵌入表示
            embed_s2c = server(embed_c2s)
            # 将服务器到客户端的嵌入表示输入到 client2 模型中，得到模型的预测结果
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)


            # # 统计预测正确的样本数。y_pred.max(1)[1] 获取预测结果中概率最大的类别索引，
            # # .eq(y) 判断预测类别索引是否与真实标签相等，.sum().item() 计算相等的数量
            # correct += y_pred.max(1)[1].eq(y).sum().item()
            # # 累加当前批次的损失
            # train_loss += loss.item()
            # # 累加当前批次的样本数
            # num_data += x.size(0)

            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            

            optimizer_server.step()

    
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
            torch.save(server.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")



def attacks_with_fc_change_label2(net1, server, net3,target_label,close_label,dataloader ,transformer,device, attack_epochs, learning_rate):


    #数据投毒攻击，翻转标签
    #两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是二
    save_dir = "./saved_model930/fc/way2"

         # 从dataloader中筛选出所有标签与target_label一致的样本
        # 重置服务器模型
    net1.reset()
    server.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net3 = copy.deepcopy(net3)
    net1.eval()
    server.eval()
    net3.eval()
    
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_net3 = torch.optim.Adam(net3.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    
    loss_function = nn.CrossEntropyLoss().to(device)

    
    for epoch in trange(attack_epochs):
        server.train()
        net3.train()
        for x, y in dataloader:
            optimizer_server.zero_grad()
            optimizer_net3.zero_grad()
            # 使用 transformer 对输入数据 x 进行预处理
            x = transformer(x)
            x, y = x.to(device), y.to(device)
            # 筛选出标签与target_label一致的样本
            mask = y == target_label
            y = y.clone()  # 创建y的副本以避免修改原始数据
            y[mask] = close_label          
            # 将输入数据 x 和真实标签 y 移动到指定的计算设备上（如 GPU）
           

            # 将输入数据 x 输入到 client1 模型中，得到客户端到服务器的嵌入表示
            embed_c2s = net1(x)
            # 将客户端到服务器的嵌入表示输入到 server 模型中，得到服务器到客户端的嵌入表示
            embed_s2c = server(embed_c2s)
            # 将服务器到客户端的嵌入表示输入到 client2 模型中，得到模型的预测结果
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)


            # # 统计预测正确的样本数。y_pred.max(1)[1] 获取预测结果中概率最大的类别索引，
            # # .eq(y) 判断预测类别索引是否与真实标签相等，.sum().item() 计算相等的数量
            # correct += y_pred.max(1)[1].eq(y).sum().item()
            # # 累加当前批次的损失
            # train_loss += loss.item()
            # # 累加当前批次的样本数
            # num_data += x.size(0)

            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            


            optimizer_server.step()
            optimizer_net3.step()

    
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
            torch.save(server.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")
            model_path = os.path.join(save_dir, f"net3_{epoch+1}.pt")
            torch.save(net3.state_dict(), model_path)
            print(f"net3模型已保存到 {model_path} 第{epoch+1}轮")
    

def train_forgery_fc_with_gmm(embeds_s2c,net3, gmm,device, attack_epochs, learning_rate , batch_size  ):
     #final_count : 指攻击所用的kmeans forgery_fc  全部只用每类别count个样本训练而成
    #os.makedirs(save_dir, exist_ok=True)
    net3.reset()
    net3.eval()
 
    optimizer_net3 = torch.optim.Adam(net3.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    
    loss_function = nn.CrossEntropyLoss().to(device)


    # base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # save_dir = os.path.join(base_dir, f"saved_model930/fc/atk_forgery_fc_way1/final_count_1/forgery_fc")
    # os.makedirs(save_dir, exist_ok=True)
    
    for epoch in trange(attack_epochs):
        total_samples = embeds_s2c.size(0)
        random_indices = torch.randperm(total_samples)  # 生成 0 到 total_samples-1 的随机排列索引  
        pointer = 0
        net3.train()
        while pointer < total_samples:
            optimizer_net3.zero_grad()

            batch_indices = random_indices[pointer:pointer + batch_size]

            if len(batch_indices) < 2:
                pointer += batch_size
                continue

            embed_s2c = embeds_s2c[batch_indices].to(device)  
            embed_s2c = embed_s2c.clone().detach()
            embed_s2c.requires_grad_(True)
            y_pred = net3(embed_s2c)
            embed_s2c_flat = torch.flatten(embed_s2c, start_dim=1).cpu().detach().numpy()
            # 从gmm模型中获取预测标签
            y = gmm.predict(embed_s2c_flat)
            y = torch.from_numpy(y).long().to(device) 
            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)

            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            optimizer_net3.step()
            pointer += batch_size

    

        # model_path = os.path.join(save_dir, f"forgery_fc_{epoch+1}.pt")
        # torch.save(net3.state_dict(), model_path)
        # print(f"net3模型已保存到 {model_path} 第{epoch+1}轮")

    return net3

def train_forgery_fc_with_dataloader(dataloader,net3,device, attack_epochs, learning_rate  ):
     #final_count : 指攻击所用的kmeans forgery_fc  全部只用每类别count个样本训练而成
    #os.makedirs(save_dir, exist_ok=True)
    net3.reset()
    net3.eval()
    optimizer_net3 = torch.optim.Adam(net3.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    loss_function = nn.CrossEntropyLoss().to(device)
    # base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # save_dir = os.path.join(base_dir, f"saved_model930/fc/atk_forgery_fc_way1/final_count_1/forgery_fc")
    # os.makedirs(save_dir, exist_ok=True)
    for _ in trange(attack_epochs):
        net3.train()
        for _, (_, batch_s2c, batch_labels) in enumerate(dataloader):
            optimizer_net3.zero_grad()
            if len(batch_labels) < 2:
                continue
            # 将numpy数组转换为PyTorch张量
            batch_s2c = torch.from_numpy(batch_s2c).float().to(device)
            batch_labels = torch.from_numpy(batch_labels).long().to(device)
            batch_s2c.requires_grad_(True)
            y_pred = net3(batch_s2c)
            loss = loss_function(y_pred, batch_labels)
            loss.backward()
            optimizer_net3.step()

        # model_path = os.path.join(save_dir, f"forgery_fc_{epoch+1}.pt")
        # torch.save(net3.state_dict(), model_path)
        # print(f"net3模型已保存到 {model_path} 第{epoch+1}轮")

    return net3


def attacks_with_forgery_fc(net1, server, net3,target_label,close_label,dataloader ,transformer,device, attack_epochs, learning_rate):

    #使用伪造的fc层进行标签翻转，采用kmeans的分类标签作为正确的分类标签
    #两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是一
    save_dir = "./saved_model930/fc/atk_forgery_fc_way1"

         # 从dataloader中筛选出所有标签与target_label一致的样本
        # 重置服务器模型
    net1.reset()
    server.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net3 = copy.deepcopy(net3)
    net1.eval()
    server.eval()
    net3.eval()
    
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    
    loss_function = nn.CrossEntropyLoss().to(device)

    save_path_kmeans = "./saved_model930/fc/kmeans_model.joblib"
    kmeans = joblib.load(save_path_kmeans)

    
    for epoch in trange(attack_epochs):
        server.train()
        for x, _ in dataloader:
            optimizer_server.zero_grad()
            net3.reset()
            # 使用 transformer 对输入数据 x 进行预处理
            x = transformer(x)
            x = x.to(device) 

            # 将输入数据 x 输入到 client1 模型中，得到客户端到服务器的嵌入表示
            embed_c2s = net1(x)
            # 将客户端到服务器的嵌入表示输入到 server 模型中，得到服务器到客户端的嵌入表示
            embed_s2c = server(embed_c2s)

            embed_s2c_flat = torch.flatten(embed_s2c, start_dim=1).cpu().detach().numpy()
            # 从kmeans模型中获取预测标签
            y = kmeans.predict(embed_s2c_flat)
            y = torch.from_numpy(y).long().to(device) 

            mask = y == target_label
            y = y.clone()  # 创建y的副本以避免修改原始数据
            y[mask] = close_label    


            # 将服务器到客户端的嵌入表示输入到 client2 模型中，得到模型的预测结果
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)


            # # 统计预测正确的样本数。y_pred.max(1)[1] 获取预测结果中概率最大的类别索引，
            # # .eq(y) 判断预测类别索引是否与真实标签相等，.sum().item() 计算相等的数量
            # correct += y_pred.max(1)[1].eq(y).sum().item()
            # # 累加当前批次的损失
            # train_loss += loss.item()
            # # 累加当前批次的样本数
            # num_data += x.size(0)

            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            

            optimizer_server.step()

    
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
            torch.save(server.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")



def attacks_with_forgery_fc2(net1, server, net3,kmeans,target_label,close_label,dataloader ,transformer,device, attack_epochs, learning_rate , ratio = 1.0):


     #使用伪造的fc层进行标签翻转，采用kmeans的分类标签作为正确的分类标签
    #两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是二



    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(base_dir, f"saved_model930/fc/atk_forgery_fc_way2/ratio{ratio}")
    os.makedirs(save_dir, exist_ok=True)

         # 从dataloader中筛选出所有标签与target_label一致的样本
        # 重置服务器模型
    net1.reset()
    server.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    server_ori = copy.deepcopy(server)

    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net3 = copy.deepcopy(net3)
    net1.eval()
    server.eval()
    server_ori.eval()
    net3.eval()
    
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_net3 = torch.optim.Adam(net3.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    
    loss_function = nn.CrossEntropyLoss().to(device)


    
    for epoch in trange(attack_epochs):
        server.train()
        net3.train()
        for x, _ in dataloader:
            optimizer_server.zero_grad()
            optimizer_net3.zero_grad()
            # 使用 transformer 对输入数据 x 进行预处理
            x = transformer(x)
            x = x.to(device)
                       
            embed_c2s = net1(x)
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s).detach()

            embed_s2c_flat = torch.flatten(embed_s2c_ori, start_dim=1).cpu().detach().numpy()
            # 从kmeans模型中获取预测标签
            y = kmeans.predict(embed_s2c_flat)
            y = torch.from_numpy(y).long().to(device) 

            # 筛选出标签与target_label一致的样本
            mask = y == target_label
            y = y.clone()  # 创建y的副本以避免修改原始数据
                      # 计算mask中True值的数量
            target_samples_count = mask.sum().item()     
            if target_samples_count > 0:
                # 设置修改比例（可以根据需要调整这个值）              
                  # 计算需要修改的样本数量
                num_to_modify = int(target_samples_count * ratio)           
                # 获取所有目标标签样本的索引
                target_indices = torch.where(mask)[0]         
                # 随机选择需要修改的样本索引
                indices_to_modify = target_indices[torch.randperm(len(target_indices))[:num_to_modify]]       
                # 只修改选中的样本
                y[indices_to_modify] = close_label     
         
            y_pred = net3(embed_s2c)
            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)



            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            


            optimizer_server.step()
            optimizer_net3.step()

    
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
            torch.save(server.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")
            model_path = os.path.join(save_dir, f"net3_{epoch+1}.pt")
            torch.save(net3.state_dict(), model_path)
            print(f"net3模型已保存到 {model_path} 第{epoch+1}轮")



def attacks_with_embeds2c_change(net1, server, net3,target_label,close_label,dataloader ,transformer,device, attack_epochs, learning_rate , save_dir):


        #把embed_s2c中标签为target_label的样本的embed替换为标签为close_label的样本的embed ，
          # 并让server以为被替换后的embed_s2c是它真正的输出，
       # 并且在进行反向传播时serve进行参数更新所用的梯度也是以“被替换后的embed_s2c是server的真正的输出”而计算得到
        #两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是二
    

         # 从dataloader中筛选出所有标签与target_label一致的样本
        # 重置服务器模型
    os.makedirs(save_dir, exist_ok=True)
    
    net1.reset()
    server.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net3 = copy.deepcopy(net3)
    net1.eval()
    server.eval()
    net3.eval()
    
    optimizer_net1 = torch.optim.Adam(net1.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_net3 = torch.optim.Adam(net3.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    
    loss_function = nn.CrossEntropyLoss().to(device)

    
    for epoch in trange(attack_epochs):
        net1.train()
        server.train()
        net3.train()
        for x, y in dataloader:
            optimizer_net1.zero_grad()
            optimizer_server.zero_grad()
            optimizer_net3.zero_grad()
            # 使用 transformer 对输入数据 x 进行预处理
            x = transformer(x)
            x, y = x.to(device), y.to(device) 
            y = y.clone()  # 创建y的副本以避免修改原始数据
            # 筛选出标签与target_label一致的样本
            mask = y == target_label
            close_mask = y == close_label 
            # 将输入数据 x 输入到 client1 模型中，得到客户端到服务器的嵌入表示
            embed_c2s = net1(x)
            # 将客户端到服务器的嵌入表示输入到 server 模型中，得到服务器到客户端的嵌入表示
            embed_s2c = server(embed_c2s)

            if torch.any(mask) and torch.any(close_mask):
                # 从close_label样本中随机选择一个embed作为替换源
                close_indices = torch.where(close_mask)[0]
                # 确保有足够的close_label样本用于替换
                if len(close_indices) > 0:
                    # 随机选择close_label样本的embed
                    selected_close_idx = torch.randint(0, len(close_indices), (torch.sum(mask).item(),))
                    close_embeds = embed_s2c[close_indices[selected_close_idx]]            
                    # 创建替换后的embed_s2c
                    server.server_intermediate[mask] = close_embeds
                    embed_s2c = server.server_intermediate.detach().requires_grad_()
                    


            
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)


            # # 统计预测正确的样本数。y_pred.max(1)[1] 获取预测结果中概率最大的类别索引，
            # # .eq(y) 判断预测类别索引是否与真实标签相等，.sum().item() 计算相等的数量
            # correct += y_pred.max(1)[1].eq(y).sum().item()
            # # 累加当前批次的损失
            # train_loss += loss.item()
            # # 累加当前批次的样本数
            # num_data += x.size(0)

            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            net1.client_backward(grad_to_client)
            

            optimizer_net1.step()
            optimizer_server.step()
            optimizer_net3.step()

    

        model_path = os.path.join(save_dir, f"server_{epoch+31}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"服务器模型已保存到 {model_path} 第{epoch+31}轮")

        model_path = os.path.join(save_dir, f"net1_{epoch+31}.pt")
        torch.save(net1.state_dict(), model_path)
        print(f"net1模型已保存到 {model_path} 第{epoch+31}轮")

        model_path = os.path.join(save_dir, f"net3_{epoch+31}.pt")
        torch.save(net3.state_dict(), model_path)
        print(f"net3模型已保存到 {model_path} 第{epoch+31}轮")


def attacks_with_embeds2c_change2(net1, server, server_ori ,net3, kmeans, target_label, dataloader, transformer, device, attack_epochs, learning_rate, save_dir):
    # 把embed_s2c中  所有  标签为target_label的样本的embed替换为随机一个标签不是target_label的样本的embed ，
    # 并让server以为被替换后的embed_s2c是它真正的输出，
    # 并且在进行反向传播时serve进行参数更新所用的梯度也是以“被替换后的embed_s2c是server的真正的输出”而计算得到
    # 两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是二
    
    # 从dataloader中筛选出所有标签与target_label一致的样本
    # 重置服务器模型
    os.makedirs(save_dir, exist_ok=True)
    
    net1.reset()
    server.reset()
    server_ori.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net3 = copy.deepcopy(net3)
    net1.eval()
    server.eval()
    net3.eval()
    server_ori.train()
    
    optimizer_net1 = torch.optim.Adam(net1.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_net3 = torch.optim.Adam(net3.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    loss_function = nn.CrossEntropyLoss().to(device)

    for epoch in trange(attack_epochs):
        net1.train()
        server.train()
        net3.train()
        for x, y in dataloader:
            optimizer_net1.zero_grad()
            optimizer_server.zero_grad()
            optimizer_net3.zero_grad()
            # 使用 transformer 对输入数据 x 进行预处理
            x = transformer(x)
            x, y = x.to(device), y.to(device) 
            
            # 将输入数据 x 输入到 client1 模型中，得到客户端到服务器的嵌入表示
            embed_c2s = net1(x)
            # 将客户端到服务器的嵌入表示输入到 server 模型中，得到服务器到客户端的嵌入表示
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s)
            #y_fc = forgery_fc(embed_s2c)

            embed_s2c_flat = torch.flatten(embed_s2c_ori, start_dim=1).cpu().detach().numpy()
            # 从kmeans模型中获取预测标签
            y_k = kmeans.predict(embed_s2c_flat)
            y_k = torch.from_numpy(y_k).long().to(device) 
            # 筛选出标签与target_label一致的样本
            target_mask = y_k == target_label
            # 筛选出所有标签不等于target_label的样本（替换源）
            non_target_mask = y_k != target_label 

            # 只有当存在目标样本和非目标样本时，才进行替换
            if torch.any(target_mask) and torch.any(non_target_mask):
                # 获取所有非目标样本的索引和对应的embed
                non_target_indices = torch.where(non_target_mask)[0]
                # 获取所有目标样本的数量
                target_count = torch.sum(target_mask).item()
                
                # 从非目标样本中随机选择与目标样本数量相等的embed（可重复选择）
                selected_non_target_idx = torch.randint(0, len(non_target_indices), (target_count,))
                non_target_embeds = embed_s2c[non_target_indices[selected_non_target_idx]]            
                
                server.server_intermediate[target_mask] = non_target_embeds
                embed_s2c = server.server_intermediate.detach().requires_grad_()
  

            # 继续后续的预测和损失计算
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)

            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            net1.client_backward(grad_to_client)
            
            optimizer_net1.step()
            optimizer_server.step()
            optimizer_net3.step()

        # 保存模型（保持原有的保存逻辑）
        model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")

        model_path = os.path.join(save_dir, f"net1_{epoch+1}.pt")
        torch.save(net1.state_dict(), model_path)
        print(f"net1模型已保存到 {model_path} 第{epoch+1}轮")

        model_path = os.path.join(save_dir, f"net3_{epoch+1}.pt")
        torch.save(net3.state_dict(), model_path)
        print(f"net3模型已保存到 {model_path} 第{epoch+1}轮")


def attacks_embeds2c_change_with_forgery_fc(net1, server, server_ori ,net3, forgery_fc, target_label, dataloader, transformer, device, attack_epochs, learning_rate, save_dir):
    #用forgery_fc取代kmeans来判断标签
    # 把embed_s2c中  所有  标签为target_label的样本的embed替换为随机一个标签不是target_label的样本的embed ，
    # 并让server以为被替换后的embed_s2c是它真正的输出，
    # 并且在进行反向传播时serve进行参数更新所用的梯度也是以“被替换后的embed_s2c是server的真正的输出”而计算得到
    # 两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是二
    
    # 从dataloader中筛选出所有标签与target_label一致的样本
    # 重置服务器模型
    os.makedirs(save_dir, exist_ok=True)
    
    net1.reset()
    server.reset()
    server_ori.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net3 = copy.deepcopy(net3)
    net1.eval()
    server.eval()
    net3.eval()
    server_ori.train()
    
    optimizer_net1 = torch.optim.Adam(net1.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_net3 = torch.optim.Adam(net3.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    loss_function = nn.CrossEntropyLoss().to(device)

    for epoch in trange(attack_epochs):
        net1.train()
        server.train()
        net3.train()
        for x, y in dataloader:
            optimizer_net1.zero_grad()
            optimizer_server.zero_grad()
            optimizer_net3.zero_grad()
            # 使用 transformer 对输入数据 x 进行预处理
            x = transformer(x)
            x, y = x.to(device), y.to(device) 
            
            # 将输入数据 x 输入到 client1 模型中，得到客户端到服务器的嵌入表示
            embed_c2s = net1(x)
            # 将客户端到服务器的嵌入表示输入到 server 模型中，得到服务器到客户端的嵌入表示
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s)
            y_fc = forgery_fc(embed_s2c_ori)

            # 筛选出标签与target_label一致的样本
            target_mask = y_fc == target_label
            # 筛选出所有标签不等于target_label的样本（替换源）
            non_target_mask = y_fc != target_label 

            # 只有当存在目标样本和非目标样本时，才进行替换
            if torch.any(target_mask) and torch.any(non_target_mask):
                # 获取所有非目标样本的索引和对应的embed
                non_target_indices = torch.where(non_target_mask)[0]
                # 获取所有目标样本的数量
                target_count = torch.sum(target_mask).item()
                
                # 从非目标样本中随机选择与目标样本数量相等的embed（可重复选择）
                selected_non_target_idx = torch.randint(0, len(non_target_indices), (target_count,))
                non_target_embeds = embed_s2c[non_target_indices[selected_non_target_idx]]            
                
                server.server_intermediate[target_mask] = non_target_embeds
                embed_s2c = server.server_intermediate.detach().requires_grad_()
  

            # 继续后续的预测和损失计算
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)

            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            net1.client_backward(grad_to_client)
            
            optimizer_net1.step()
            optimizer_server.step()
            optimizer_net3.step()

        # 保存模型（保持原有的保存逻辑）
        model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")

        model_path = os.path.join(save_dir, f"net1_{epoch+1}.pt")
        torch.save(net1.state_dict(), model_path)
        print(f"net1模型已保存到 {model_path} 第{epoch+1}轮")

        model_path = os.path.join(save_dir, f"net3_{epoch+1}.pt")
        torch.save(net3.state_dict(), model_path)
        print(f"net3模型已保存到 {model_path} 第{epoch+1}轮")


def attacks_with_embeds2c_change3(net1, server, net3, target_label, dataloader, transformer, device, attack_epochs, learning_rate, save_dir, ratio=0.5):
    # 把embed_s2c中比例为ratio的标签为target_label的样本的embed替换为随机一个标签不是target_label的样本的embed ，
    # 并让server以为被替换后的embed_s2c是它真正的输出，
    # 并且在进行反向传播时serve进行参数更新所用的梯度也是以“被替换后的embed_s2c是server的真正的输出”而计算得到
    # 两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是二
    
    # 校验ratio参数合法性
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"ratio参数必须在[0.0, 1.0]之间，当前输入为{ratio}")
    
    # 从dataloader中筛选出所有标签与target_label一致的样本
    # 重置服务器模型
    os.makedirs(save_dir, exist_ok=True)
    
    net1.reset()
    server.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net3 = copy.deepcopy(net3)
    net1.eval()
    server.eval()
    net3.eval()
    
    optimizer_net1 = torch.optim.Adam(net1.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_net3 = torch.optim.Adam(net3.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    loss_function = nn.CrossEntropyLoss().to(device)

    for epoch in trange(attack_epochs):
        net1.train()
        server.train()
        net3.train()
        for x, y in dataloader:
            optimizer_net1.zero_grad()
            optimizer_server.zero_grad()
            optimizer_net3.zero_grad()
            # 使用 transformer 对输入数据 x 进行预处理
            x = transformer(x)
            x, y = x.to(device), y.to(device) 
            y = y.clone()  # 创建y的副本以避免修改原始数据
            # 筛选出标签与target_label一致的样本（所有target样本）
            target_mask = y == target_label
            # 筛选出所有标签不等于target_label的样本（替换源）
            non_target_mask = y != target_label 
            
            # 将输入数据 x 输入到 client1 模型中，得到客户端到服务器的嵌入表示
            embed_c2s = net1(x)
            # 将客户端到服务器的嵌入表示输入到 server 模型中，得到服务器到客户端的嵌入表示
            embed_s2c = server(embed_c2s)
            

            # 只有当存在目标样本、非目标样本，且ratio>0时，才进行替换
            if ratio > 0.0 and torch.any(target_mask) and torch.any(non_target_mask):
                # 获取所有target样本的索引
                target_indices = torch.where(target_mask)[0]
                # 获取所有non-target样本的索引
                non_target_indices = torch.where(non_target_mask)[0]
                
                # 计算需要替换的target样本数量（按ratio比例）
                total_target = len(target_indices)
                replace_count = int(total_target * ratio)
                
                # 如果需要替换的数量>0，才执行替换（避免ratio=0或total_target=0的情况）
                if replace_count > 0:
                    # 从所有target样本索引中，随机挑选replace_count个样本作为“要替换的target样本”
                    # permute打乱索引后取前replace_count个
                    selected_target_idx = target_indices[torch.randperm(total_target)[:replace_count]]
                    
                    # 从non-target样本中，随机挑选replace_count个embed（可重复选择）
                    selected_non_target_rand_idx = torch.randint(0, len(non_target_indices), (replace_count,))
                    selected_non_target_embeds = embed_s2c[non_target_indices[selected_non_target_rand_idx]]
                    
                    # 只替换“选中的target样本”的embed，未选中的target样本保持原始embed不变
                    server.server_intermediate[selected_target_idx] = selected_non_target_embeds
                    embed_s2c = server.server_intermediate.detach().requires_grad_()

            # 继续后续的预测和损失计算
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)

            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            net1.client_backward(grad_to_client)
            
            optimizer_net1.step()
            optimizer_server.step()
            optimizer_net3.step()

        # 保存模型（保持原有的保存逻辑）
        model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")

        model_path = os.path.join(save_dir, f"net1_{epoch+1}.pt")
        torch.save(net1.state_dict(), model_path)
        print(f"net1模型已保存到 {model_path} 第{epoch+1}轮")

        model_path = os.path.join(save_dir, f"net3_{epoch+1}.pt")
        torch.save(net3.state_dict(), model_path)
        print(f"net3模型已保存到 {model_path} 第{epoch+1}轮")

def attacks_with_embeds2c_change4(net1, server, server_ori ,net3, kmeans, target_label_list, dataloader, transformer, device, attack_epochs, learning_rate, save_dir):
    # 把embed_s2c中  所有  标签为LIST:target_label的样本的embed替换为随机一个标签不是LIST:target_label的样本的embed ，
    # 并让server以为被替换后的embed_s2c是它真正的输出，
    # 并且在进行反向传播时serve进行参数更新所用的梯度也是以“被替换后的embed_s2c是server的真正的输出”而计算得到
    # 两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是二
    
    # 从dataloader中筛选出所有标签与target_label一致的样本
    # 重置服务器模型
    os.makedirs(save_dir, exist_ok=True)
    
    net1.reset()
    server.reset()
    server_ori.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net3 = copy.deepcopy(net3)
    net1.eval()
    server.eval()
    net3.eval()
    server_ori.train()
    
    optimizer_net1 = torch.optim.Adam(net1.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_net3 = torch.optim.Adam(net3.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    loss_function = nn.CrossEntropyLoss().to(device)

    # 将target_label_set转换为tensor，方便后续判断（确保在指定设备上）
    target_label_tensor = torch.tensor(target_label_list, device=device)

    for epoch in trange(attack_epochs):
        net1.train()
        server.train()
        net3.train()
        for x, y in dataloader:
            optimizer_net1.zero_grad()
            optimizer_server.zero_grad()
            optimizer_net3.zero_grad()
            # 使用 transformer 对输入数据 x 进行预处理
            x = transformer(x)
            x, y = x.to(device), y.to(device) 
            
            # 将输入数据 x 输入到 client1 模型中，得到客户端到服务器的嵌入表示
            embed_c2s = net1(x)
            # 将客户端到服务器的嵌入表示输入到 server 模型中，得到服务器到客户端的嵌入表示
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s)
            #y_fc = forgery_fc(embed_s2c)
            embed_s2c_flat = torch.flatten(embed_s2c_ori, start_dim=1).cpu().detach().numpy()
            #从kmeans模型中获取预测标签
            y_k = kmeans.predict(embed_s2c_flat)
            y_k = torch.from_numpy(y_k).long().to(device) 
            
            # 关键修改1：筛选出标签属于target_label_set的样本（支持多标签）
            # 使用torch.isin判断y_k是否在目标标签集合中
            target_mask = torch.isin(y_k, target_label_tensor)
            # 关键修改2：筛选出所有标签不属于target_label_set的样本（替换源）
            non_target_mask = ~torch.isin(y_k, target_label_tensor)

            # 只有当存在目标样本和非目标样本时，才进行替换
            if torch.any(target_mask) and torch.any(non_target_mask):
                # 获取所有非目标样本的索引和对应的embed
                non_target_indices = torch.where(non_target_mask)[0]
                # 获取所有目标样本的数量
                target_count = torch.sum(target_mask).item()
                
                # 从非目标样本中随机选择与目标样本数量相等的embed（可重复选择）
                selected_non_target_idx = torch.randint(0, len(non_target_indices), (target_count,))
                non_target_embeds = embed_s2c[non_target_indices[selected_non_target_idx]]            
                
                server.server_intermediate[target_mask] = non_target_embeds
                embed_s2c = server.server_intermediate.detach().requires_grad_()
  

            # 继续后续的预测和损失计算
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)

            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            net1.client_backward(grad_to_client)
            
            optimizer_net1.step()
            optimizer_server.step()
            optimizer_net3.step()

        # 保存模型（保持原有的保存逻辑）
        model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")

        model_path = os.path.join(save_dir, f"net1_{epoch+1}.pt")
        torch.save(net1.state_dict(), model_path)
        print(f"net1模型已保存到 {model_path} 第{epoch+1}轮")

        model_path = os.path.join(save_dir, f"net3_{epoch+1}.pt")
        torch.save(net3.state_dict(), model_path)
        print(f"net3模型已保存到 {model_path} 第{epoch+1}轮")




def attacks_final1(embeds_c2s, server, net3,kmeans,target_label, close_label ,device, attack_epochs, learning_rate,batch_size,save_dir):

    #使用伪造的fc层进行标签翻转，采用kmeans的分类标签作为正确的分类标签
    #两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是一

    #final_count : 指攻击所用的kmeans forgery_fc  全部只用每类别count个样本训练而成
    #target_count : 指攻击所用的kmeans forgery_fc用全部样本训练 ，但是攻击的时候target类别只有count个样本， 其余类别有全部样本

    
    os.makedirs(save_dir, exist_ok=True)

    server.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    server_ori = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net3 = copy.deepcopy(net3)
    server.eval()
    server_ori.eval()
    net3.eval() 
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)   
    loss_function = nn.CrossEntropyLoss().to(device)
    
    for epoch in trange(attack_epochs):
        total_samples = embeds_c2s.size(0)
        random_indices = torch.randperm(total_samples)  # 生成 0 到 total_samples-1 的随机排列索引  
        pointer = 0
        server.train()

        while pointer < total_samples:
            optimizer_server.zero_grad()
            net3.reset()
            # 按随机索引截取当前批次的索引（而非原始顺序）
            batch_indices = random_indices[pointer:pointer + batch_size]
            
            # 确保批次大小至少为2，避免BatchNorm层错误
            if len(batch_indices) < 2:
                pointer += batch_size
                continue
                
        # 根据随机索引获取当前批次数据，并移动到设备
            embed_c2s = embeds_c2s[batch_indices].to(device)  # 核心：用随机索引取数据
            embed_c2s = embed_c2s.clone().detach()
            embed_c2s.requires_grad_(True)
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s).detach()

            embed_s2c_flat = torch.flatten(embed_s2c_ori, start_dim=1).cpu().detach().numpy()
            
            # 从kmeans模型中获取预测标签
            y = kmeans.predict(embed_s2c_flat)
            y = torch.from_numpy(y).long().to(device) 

            mask = y == target_label
            y = y.clone()  # 创建y的副本以避免修改原始数据
            y[mask] = close_label    


            # 将服务器到客户端的嵌入表示输入到 client2 模型中，得到模型的预测结果
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)


            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            

            optimizer_server.step()
            pointer += batch_size
    
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        # if (epoch<20) or ((epoch+1) % 10 == 0):
        #     model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        #     torch.save(server.state_dict(), model_path)
        #     print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")

        # if (epoch>40 and epoch<50) or ((epoch+1) % 10 == 0):
        #     model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        #     torch.save(server.state_dict(), model_path)
        #     print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")
        
        model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")





def attacks_final2(embeds_c2s, server, net3,kmeans,target_label,close_label,device, attack_epochs, learning_rate,batch_size , count):


     #使用伪造的fc层进行标签翻转，采用kmeans的分类标签作为正确的分类标签
    #两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是二
    # 获取项目根目录（假设当前文件在src目录下）
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    save_dir = os.path.join(base_dir, f"saved_model930/fc/atk_forgery_fc_way2/fix_count_{count}")
    os.makedirs(save_dir, exist_ok=True)

         # 从dataloader中筛选出所有标签与target_label一致的样本
        # 重置服务器模型
    server.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    server_ori = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net3 = copy.deepcopy(net3)
    server.eval()
    server_ori.eval()
    net3.eval()
    
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_net3 = torch.optim.Adam(net3.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    
    loss_function = nn.CrossEntropyLoss().to(device)


    for epoch in trange(attack_epochs):
        total_samples = embeds_c2s.size(0)
        random_indices = torch.randperm(total_samples)  # 生成 0 到 total_samples-1 的随机排列索引  
        pointer = 0
        server.train()
        net3.train()
        while pointer < total_samples:
            optimizer_server.zero_grad()
            optimizer_net3.zero_grad()
            # 按随机索引截取当前批次的索引（而非原始顺序）
            batch_indices = random_indices[pointer:pointer + batch_size]
        # 根据随机索引获取当前批次数据，并移动到设备
            embed_c2s = embeds_c2s[batch_indices].to(device)  # 核心：用随机索引取数据
            embed_c2s = embed_c2s.clone().detach()
            embed_c2s.requires_grad_(True)
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s).detach()

            embed_s2c_flat = torch.flatten(embed_s2c_ori, start_dim=1).cpu().detach().numpy()
            # 从kmeans模型中获取预测标签
            y = kmeans.predict(embed_s2c_flat)
            y = torch.from_numpy(y).long().to(device) 

            # 筛选出标签与target_label一致的样本
            mask = y == target_label
            y = y.clone()  # 创建y的副本以避免修改原始数据
                      # 计算mask中True值的数量  
            y[mask] = close_label
            y_pred = net3(embed_s2c)
            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)



            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            


            optimizer_server.step()
            optimizer_net3.step()
            pointer += batch_size

    
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
            torch.save(server.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")
            model_path = os.path.join(save_dir, f"net3_{epoch+1}.pt")
            torch.save(net3.state_dict(), model_path)
            print(f"net3模型已保存到 {model_path} 第{epoch+1}轮")


def attacks_final3(embeds_c2s, server, net3, kmeans, target_label, device, attack_epochs, learning_rate, batch_size, save_dir, num_classes):

    #使用伪造的fc层进行标签翻转，采用kmeans的分类标签作为正确的分类标签
    #将target_label类别样本的标签随机翻转为非target_label类别的标签
    #两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是一

    os.makedirs(save_dir, exist_ok=True)

    server.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    server_ori = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net3 = copy.deepcopy(net3)
    server.eval()
    server_ori.eval()
    net3.eval() 
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)   
    loss_function = nn.CrossEntropyLoss().to(device)
    
    # 生成所有非target_label的类别列表（提前计算，避免重复生成）
    non_target_labels = [label for label in range(num_classes) if label != target_label]
    # 转换为tensor方便后续处理
    non_target_labels_tensor = torch.tensor(non_target_labels, dtype=torch.long, device=device)
    
    for epoch in trange(attack_epochs):
        total_samples = embeds_c2s.size(0)
        random_indices = torch.randperm(total_samples)  # 生成 0 到 total_samples-1 的随机排列索引  
        pointer = 0
        server.train()

        while pointer < total_samples:
            optimizer_server.zero_grad()
            net3.reset()
            # 按随机索引截取当前批次的索引（而非原始顺序）
            batch_indices = random_indices[pointer:pointer + batch_size]
            
            # 确保批次大小至少为2，避免BatchNorm层错误
            if len(batch_indices) < 2:
                pointer += batch_size
                continue
                
            # 根据随机索引获取当前批次数据，并移动到设备
            embed_c2s = embeds_c2s[batch_indices].to(device)  # 核心：用随机索引取数据
            embed_c2s = embed_c2s.clone().detach()
            embed_c2s.requires_grad_(True)
            embed_s2c = server(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s).detach()

            embed_s2c_flat = torch.flatten(embed_s2c_ori, start_dim=1).cpu().detach().numpy()

            y = kmeans.predict(embed_s2c_flat)
            y = torch.from_numpy(y).long().to(device) 

            # ========== 核心修改部分 ==========
            mask = y == target_label
            y = y.clone()  # 创建y的副本以避免修改原始数据
            
            # 找到所有需要翻转标签的样本索引
            target_indices = torch.nonzero(mask, as_tuple=True)[0]
            
            if len(target_indices) > 0:
                # 为每个目标样本随机选择一个非target_label的标签
                # 生成随机索引，范围是non_target_labels的长度
                random_non_target_indices = torch.randint(
                    low=0, 
                    high=len(non_target_labels), 
                    size=(len(target_indices),), 
                    device=device
                )
                # 获取对应的随机非目标标签
                random_non_target_labels = non_target_labels_tensor[random_non_target_indices]
                # 替换目标样本的标签
                y[target_indices] = random_non_target_labels
            # ========== 核心修改部分结束 ==========


            # 将服务器到客户端的嵌入表示输入到 client2 模型中，得到模型的预测结果
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)


            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            

            optimizer_server.step()
            pointer += batch_size
    
        #保存服务器模型
        model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")


def attacks_dataloader1(dataloader, server, target_label, close_label, 
                        device, attack_epochs, learning_rate, save_dir,
                        embeds_s2c_all, labels_all):

    
    os.makedirs(save_dir, exist_ok=True)

        # VGG 结构比较直筒，特征传导更直接，alpha 可以稍微大一点保持稳定，或者保持 5.0
    alpha = 0.5

    print(f"启动 VGG16 外科手术式攻击...")
    print(f"策略: 冻结前端卷积块, 只微调最后一个卷积块 | 攻击方式: 质心牵引")

    # ==========================================
    # 1. 计算引导质心 (Guided Vector) - 逻辑不变
    # ==========================================
    print("正在计算引导质心...")
    close_indices = np.where(labels_all == close_label)[0]
    if len(close_indices) == 0:
        raise ValueError(f"错误: 数据集中找不到类别 {close_label} 的样本")
    
    sample_size = min(len(close_indices), 1000)
    sampled_indices = np.random.choice(close_indices, sample_size, replace=False)
    
    target_vectors = embeds_s2c_all[sampled_indices] 
    guided_centroid = np.mean(target_vectors, axis=0)
    guided_centroid = torch.from_numpy(guided_centroid).float().to(device)
    guided_centroid = guided_centroid.unsqueeze(0)
    print(f"引导质心计算完成。Shape: {guided_centroid.shape}")

    # ==========================================
    # 2. 模型准备与 VGG 特有的层级冻结
    # ==========================================
    server.reset()
    
    # 准备锚点模型
    anchor_server = copy.deepcopy(server)
    anchor_server.eval()
    for p in anchor_server.parameters(): p.requires_grad = False
    
    # --- VGG 核心修改：基于 Block 的冻结策略 ---
    
    # 1. 首先冻结 Server 的所有参数
    for param in server.parameters():
        param.requires_grad = False
        
    # 2. 找到需要解冻的“最后几层”
    # 在你的 NetServer 代码中，self.layer1 是一个 Sequential，
    # 它的子元素 (children) 是 make_layer_vgg16 生成的卷积块 (Sequential)
    # 最后一个子元素通常是 AdaptiveAvgPool2d，倒数第二个是最后一个卷积块
    
    # 获取 layer1 的所有子模块列表
    server_blocks = list(server.layer1.children())
    total_blocks = len(server_blocks)
    
    # 策略：解冻最后 2 个模块 (包含最后一个 ConvBlock 和 AvgPool)
    # 这相当于 ResNet 的 Layer4
    unfreeze_count = 2 
    
    print(f"VGG结构分析: NetServer 共有 {total_blocks} 个模块。正在解冻最后 {unfreeze_count} 个模块...")

    # 遍历最后几个模块进行解冻
    for i in range(1, unfreeze_count + 1):
        target_block = server_blocks[-i] # 倒数第 i 个块
        
        # 解冻参数
        for param in target_block.parameters():
            param.requires_grad = True
            
        # 设置为训练模式 (允许 Dropout 生效，如果有的话)
        target_block.train()
        
        # 依然冻结 BN 层 (虽然 VGG 某些变体没有 BN，但加上这句更稳健)
        for m in target_block.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    # ==========================================
    # 3. 优化器与损失
    # ==========================================
    # 过滤出解冻的参数
    trainable_params = [p for p in server.parameters() if p.requires_grad]
    print(f"可训练参数组数量: {len(trainable_params)}")
    
    if len(trainable_params) == 0:
        raise ValueError("警告：没有参数被解冻！请检查 unfreeze_count 设置。")

    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)
    loss_mse = nn.MSELoss()
    

    # ==========================================
    # 4. 攻击循环 (使用 forward2)
    # ==========================================
    for epoch in trange(attack_epochs):
        total_loss = 0
        atk_loss_sum = 0
        stable_loss_sum = 0
        batches = 0

        for _, (batch_c2s, _, batch_labels) in enumerate(dataloader):
            if len(batch_labels) < 2: continue

            batch_c2s = torch.from_numpy(batch_c2s).float().to(device)
            batch_labels = torch.from_numpy(batch_labels).long().to(device)
            batch_c2s.requires_grad_(False) # Client 端不需要梯度
            
            optimizer.zero_grad()
            
            # --- 关键：调用 VGG 的 forward2 ---
            embed_current = server.forward2(batch_c2s)
            
            with torch.no_grad():
                # Anchor 使用同样的 forward2
                embed_anchor = anchor_server(batch_c2s)
            
            # Masks
            target_mask = (batch_labels == target_label)
            non_target_mask = ~target_mask
            
            loss = 0
            
            # Loss A: 攻击损失 (MSE 拉向质心)
            if target_mask.sum() > 0:
                target_features = embed_current[target_mask]
                current_batch_centroid = guided_centroid.expand(target_features.size(0), -1, -1, -1)
                l_atk = loss_mse(target_features, current_batch_centroid)
                loss += l_atk
                atk_loss_sum += l_atk.item()
            
            # Loss B: 稳定性损失 (MSE 锁死锚点)
            if non_target_mask.sum() > 0:
                stable_features = embed_current[non_target_mask]
                anchor_features = embed_anchor[non_target_mask]
                l_stable = loss_mse(stable_features, anchor_features)
                loss += alpha * l_stable
                stable_loss_sum += (alpha * l_stable).item()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss / batches if batches > 0 else 0
        avg_atk = atk_loss_sum / batches if batches > 0 else 0
        avg_stable = stable_loss_sum / batches if batches > 0 else 0
        
        print(f"Epoch {epoch+1}/{attack_epochs} | Total: {avg_loss:.4f} | Atk: {avg_atk:.4f} | Stable: {avg_stable:.4f}")

        model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"模型已保存: {model_path}")


def attacks_dataloader2(dataloader, server, target_label, close_label, 
                        device, attack_epochs, learning_rate, save_dir,
                        embeds_s2c_all, labels_all): # 新增两个参数用于计算质心

    #无net3 用质心牵引去攻击 不冻结 Layer1-3 ， 微调server所有层
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"启动外科手术式攻击 (Surgical Attack)...")
    print(f"策略: 冻结 Layer1-3, 只微调 Layer4 | 攻击方式: 质心牵引 (Centroid Pulling)")

    # ==========================================
    # 1. 计算引导质心 (Guided Vector)
    # ==========================================
    # 从所有缓存的特征中，找到 Close Label (原本想让目标类变成的那个类) 的所有样本
    # 计算它们的平均值，这就是我们要把 Target Label 拖过去的目的地
    print("正在计算引导质心...")
    # 假设 labels_all 是 numpy array
    close_indices = np.where(labels_all == close_label)[0]
    if len(close_indices) == 0:
        raise ValueError(f"错误: 数据集中找不到类别 {close_label} 的样本，无法计算质心！")
    
    # 随机采样一部分计算均值即可，避免内存爆炸
    sample_size = min(len(close_indices), 1000)
    sampled_indices = np.random.choice(close_indices, sample_size, replace=False)
    
    # 获取特征向量并转为 Tensor
    target_vectors = embeds_s2c_all[sampled_indices] # shape: [N, C, H, W]
    # 计算均值向量 (Centroid)
    guided_centroid = np.mean(target_vectors, axis=0)
    guided_centroid = torch.from_numpy(guided_centroid).float().to(device)
    # 增加 batch 维度以便广播: [1, C, H, W]
    guided_centroid = guided_centroid.unsqueeze(0)
    print(f"引导质心计算完成。Shape: {guided_centroid.shape}")

    # ==========================================
    # 2. 模型准备与层级冻结 (Layer Freezing)
    # ==========================================
    server.reset()
    
    # 准备锚点模型 (用于保持非目标类稳定)
    anchor_server = copy.deepcopy(server)
    anchor_server.eval()
    for p in anchor_server.parameters(): p.requires_grad = False
    
    # 冻结所有 BatchNorm 层 (防止统计量漂移)
    server.train()# 开启 eval 模式会让 BN 使用全局统计量，不更新 running_mean/var
    # 但是我们需要 dropout (如果有) 或者梯度更新，所以手动开启 layer4 的训练模式
    # ResNet/ResNeXt 的 layer4 主要是卷积和 BN
    # 强制将 Layer4 设置为训练模式，但保持 BN 为 eval
    optimizer = torch.optim.Adam(server.parameters(), lr=learning_rate , weight_decay=1e-4)
    # ==========================================
    # 3. 优化器与损失
    # ==========================================
    # 过滤出需要更新的参数 (只有 layer4)
    # 学习率可以适当调大一点，因为我们只动最后一层，且 MSE 梯度通常较小
    loss_mse = nn.MSELoss()
    # 稳定性权重
    alpha = 10.0 

    # ==========================================
    # 4. 攻击循环
    # ==========================================
    for epoch in trange(attack_epochs):
        server.train()
        for m in server.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        total_loss = 0
        atk_loss_sum = 0
        stable_loss_sum = 0
        batches = 0

        for _, (batch_c2s, _, batch_labels) in enumerate(dataloader):
            if len(batch_labels) < 2: continue

            batch_c2s = torch.from_numpy(batch_c2s).float().to(device)
            batch_labels = torch.from_numpy(batch_labels).long().to(device)
            batch_c2s.requires_grad_(True)
            
            optimizer.zero_grad()
            
            # Forward
            embed_current = server.forward2(batch_c2s)
            with torch.no_grad():
                embed_anchor = anchor_server(batch_c2s)
            
            # Masks
            target_mask = (batch_labels == target_label)
            non_target_mask = ~target_mask
            
            loss = 0
            
            # --- Loss A: 攻击损失 (Centroid Pulling) ---
            # 强迫 Target 样本的特征变成 Close Label 的质心
            if target_mask.sum() > 0:
                target_features = embed_current[target_mask]
                # 扩展质心以匹配 batch 大小
                current_batch_centroid = guided_centroid.expand(target_features.size(0), -1, -1, -1)
                
                # 直接计算 MSE：让特征通过欧氏距离靠近质心
                l_atk = loss_mse(target_features, current_batch_centroid)
                loss += l_atk
                atk_loss_sum += l_atk.item()
            
            # --- Loss B: 稳定性损失 (Anchor Locking) ---
            # 强迫非 Target 样本保持原样
            if non_target_mask.sum() > 0:
                stable_features = embed_current[non_target_mask]
                anchor_features = embed_anchor[non_target_mask]
                l_stable = loss_mse(stable_features, anchor_features)
                
                loss += alpha * l_stable
                stable_loss_sum += (alpha * l_stable).item()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        # 打印日志
        avg_loss = total_loss / batches if batches > 0 else 0
        avg_atk = atk_loss_sum / batches if batches > 0 else 0
        avg_stable = stable_loss_sum / batches if batches > 0 else 0
        
        print(f"Epoch {epoch+1}/{attack_epochs} | Total: {avg_loss:.4f} | Atk(MSE): {avg_atk:.4f} | Stable: {avg_stable:.4f}")

        # 保存
        model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"模型已保存: {model_path}")


def attacks_dataloader3(dataloader, server, target_label, close_label, 
                        device, attack_epochs, learning_rate, save_dir,
                        embeds_s2c_all, labels_all):
    #attacks_dataloader1取消稳定性损失版
    os.makedirs(save_dir, exist_ok=True)
    print(f"启动 VGG16 外科手术式攻击...")
    print(f"策略: 冻结前端卷积块, 只微调最后一个卷积块 | 攻击方式: 质心牵引")

    # ==========================================
    # 1. 计算引导质心 (Guided Vector) - 逻辑不变
    # ==========================================
    print("正在计算引导质心...")
    close_indices = np.where(labels_all == close_label)[0]
    if len(close_indices) == 0:
        raise ValueError(f"错误: 数据集中找不到类别 {close_label} 的样本")
    
    sample_size = min(len(close_indices), 1000)
    sampled_indices = np.random.choice(close_indices, sample_size, replace=False) 
    target_vectors = embeds_s2c_all[sampled_indices] 
    guided_centroid = np.mean(target_vectors, axis=0)
    guided_centroid = torch.from_numpy(guided_centroid).float().to(device)
    guided_centroid = guided_centroid.unsqueeze(0)
    print(f"引导质心计算完成。Shape: {guided_centroid.shape}")

    # ==========================================
    # 2. 模型准备与 VGG 特有的层级冻结
    # ==========================================
    server.reset()

    # --- VGG 核心修改：基于 Block 的冻结策略 ---
    
    # 1. 首先冻结 Server 的所有参数
    for param in server.parameters():
        param.requires_grad = False
        
    # 2. 找到需要解冻的“最后几层”
    # 在你的 NetServer 代码中，self.layer1 是一个 Sequential，
    # 它的子元素 (children) 是 make_layer_vgg16 生成的卷积块 (Sequential)
    # 最后一个子元素通常是 AdaptiveAvgPool2d，倒数第二个是最后一个卷积块
    
    # 获取 layer1 的所有子模块列表
    server_blocks = list(server.layer1.children())
    total_blocks = len(server_blocks)
    
    # 策略：解冻最后 2 个模块 (包含最后一个 ConvBlock 和 AvgPool)
    # 这相当于 ResNet 的 Layer4
    unfreeze_count = 2  
    print(f"VGG结构分析: NetServer 共有 {total_blocks} 个模块。正在解冻最后 {unfreeze_count} 个模块...")

    # 遍历最后几个模块进行解冻
    for i in range(1, unfreeze_count + 1):
        target_block = server_blocks[-i] # 倒数第 i 个块
        
        # 解冻参数
        for param in target_block.parameters():
            param.requires_grad = True
            
        # 设置为训练模式 (允许 Dropout 生效，如果有的话)
        target_block.train()
        
        # 依然冻结 BN 层 (虽然 VGG 某些变体没有 BN，但加上这句更稳健)
        for m in target_block.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    # ==========================================
    # 3. 优化器与损失
    # ==========================================
    # 过滤出解冻的参数
    trainable_params = [p for p in server.parameters() if p.requires_grad]
    print(f"可训练参数组数量: {len(trainable_params)}")
    
    if len(trainable_params) == 0:
        raise ValueError("警告：没有参数被解冻！请检查 unfreeze_count 设置。")

    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)
    loss_mse = nn.MSELoss()
    
    # ==========================================
    # 4. 攻击循环 (使用 forward2)
    # ==========================================
    for epoch in trange(attack_epochs):
        total_loss = 0
        atk_loss_sum = 0
        stable_loss_sum = 0
        batches = 0

        for _, (batch_c2s, _, batch_labels) in enumerate(dataloader):
            if len(batch_labels) < 2: continue
            batch_c2s = torch.from_numpy(batch_c2s).float().to(device)
            batch_labels = torch.from_numpy(batch_labels).long().to(device)
            batch_c2s.requires_grad_(False) # Client 端不需要梯度
            optimizer.zero_grad()   
            # --- 关键：调用 VGG 的 forward2 ---
            embed_current = server.forward2(batch_c2s)
   
            # Masks
            target_mask = (batch_labels == target_label)
            non_target_mask = ~target_mask
            
            loss = 0
            
            # Loss A: 攻击损失 (MSE 拉向质心)
            if target_mask.sum() > 0:
                target_features = embed_current[target_mask]
                current_batch_centroid = guided_centroid.expand(target_features.size(0), -1, -1, -1)
                l_atk = loss_mse(target_features, current_batch_centroid)
                loss += l_atk
                atk_loss_sum += l_atk.item()
    
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            batches += 1    

        avg_loss = total_loss / batches if batches > 0 else 0
        avg_atk = atk_loss_sum / batches if batches > 0 else 0
        avg_stable = stable_loss_sum / batches if batches > 0 else 0
        
        print(f"Epoch {epoch+1}/{attack_epochs} | Total: {avg_loss:.4f} | Atk: {avg_atk:.4f} | Stable: {avg_stable:.4f}")

        model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"模型已保存: {model_path}")


def attacks_dataloader_ratio1(dataloader, server, net3,  target_label, close_label ,device, attack_epochs, learning_rate,  save_dir , ratio):

    """
    使用mydataloader进行攻击训练，用伪造的fc层进行标签翻转
    将target_label按指定比例(ratio)随机翻转为close_label
    
    参数新增：
    ratio: float, 0~1之间的数值，表示要翻转标签的样本比例
    """
    # 设置随机种子确保可复现性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    #使用mydataloader进行攻击训练，用伪造的fc层进行标签翻转，将target_label固定翻转为close_label
    #两种办法：一是只训练server不训练fc，二是训练server和fc ，这里是一

    os.makedirs(save_dir, exist_ok=True)
    server.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net3 = copy.deepcopy(net3)
    server.eval()
    net3.eval() 
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)   
    loss_function = nn.CrossEntropyLoss().to(device)
    
    for epoch in trange(attack_epochs):
        server.train()

        for _, (batch_c2s, _, batch_labels) in enumerate(dataloader):
            optimizer_server.zero_grad()
            net3.reset()
            # 确保批次大小至少为2，避免BatchNorm层错误
            if len(batch_labels) < 2:
                continue   

            # 将numpy数组转换为PyTorch张量
            batch_c2s = torch.from_numpy(batch_c2s).float().to(device)
            batch_labels = torch.from_numpy(batch_labels).long().to(device)
            # 根据随机索引获取当前批次数据，并移动到设备
            batch_c2s.requires_grad_(True)
            embed_s2c = server(batch_c2s)

            # ========== 核心修改部分 ==========
            # 1. 找到所有target_label的位置
            mask_all_target = batch_labels == target_label
            
            # 2. 获取target_label样本的索引
            target_indices = torch.nonzero(mask_all_target).squeeze(dim=1)
            
            # 3. 如果有target_label样本，按比例随机选择部分样本
            if len(target_indices) > 0:
                # 计算要选择的样本数量
                num_to_flip = max(0, min(len(target_indices), int(np.ceil(len(target_indices) * ratio))))
                
                # 4. 随机选择指定比例的索引
                if num_to_flip > 0:
                    # 随机打乱索引
                    permuted_indices = torch.randperm(len(target_indices))
                    # 选择前num_to_flip个索引
                    selected_indices = target_indices[permuted_indices[:num_to_flip]]
                    
                    # 5. 创建新的mask，只标记选中的样本
                    mask = torch.zeros_like(mask_all_target, dtype=torch.bool)
                    mask[selected_indices] = True
                else:
                    # 比例为0时，mask全为False
                    mask = torch.zeros_like(mask_all_target, dtype=torch.bool)
            else:
                # 没有target_label样本时，mask全为False
                mask = torch.zeros_like(mask_all_target, dtype=torch.bool)

            # 6. 标签翻转（只翻转选中的样本）
            y = batch_labels.clone()  # 创建y的副本以避免修改原始数据
            y[mask] = close_label 

            # ========== 核心修改结束 ==========

            # 将服务器到客户端的嵌入表示输入到 client2 模型中，得到模型的预测结果
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)

            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            
            optimizer_server.step()

        #保存服务器模型
        model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")


def get_embeds_c2s_last_epoch(client1, server, client2,
                            dataloader, transformer, optimizers, loss_function, device,kmeans,target_label,count):

    #只收集目标样本的embeds_c2s  ,  数量由count指定

    # 初始化客户端到服务器的嵌入表示、服务器到客户端的嵌入表示和标签为 None
    embeds_c2s = None

    # 将 client1、server 和 client2 模型设置为训练模式
    client1.train()
    server.train()
    client2.train()

    collected = 0   #记录实际收集到的目标标签样本数
    total_target_samples = 0  #记录数据集中的目标标签样本数
    # 标记是否需要打印提示信息
    

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
        embed_s2c_flat = torch.flatten(embed_s2c, start_dim=1) 
        embed_s2c_flat = embed_s2c_flat.detach().cpu().numpy()
        y_kmeans = kmeans.predict(embed_s2c_flat)
        y_kmeans = torch.from_numpy(y_kmeans).long().to(device) 
        # 将服务器到客户端的嵌入表示输入到 client2 模型中，得到模型的预测结果
        y_pred = client2(embed_s2c)
        # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
        loss = loss_function(y_pred, y)

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
        
        # 1. 筛选当前批次中标签为target_label的样本索引
        target_mask = (y_kmeans == target_label)
        current_target_embeds = embed_c2s[target_mask]
        current_target_count = current_target_embeds.size(0)
        total_target_samples += current_target_count

        # 2. 确定当前批次需要收集的数量
        if count < 0:  # 负数表示收集所有目标样本
            need_collect = current_target_count
        else:  # 正数时不超过剩余需要收集的数量
            need_collect = min(current_target_count, count - collected)

        # 3. 收集嵌入表示（如果还有需要收集的数量且当前批次有符合条件的样本）
        if need_collect > 0 :
            # 截取需要的数量
            selected_embeds = current_target_embeds[:need_collect].cpu()
            
            if embeds_c2s is None:
                embeds_c2s = selected_embeds
            else:
                embeds_c2s = torch.cat((embeds_c2s, selected_embeds), dim=0)
            
            collected += need_collect

        # 训练结束后处理
    else:
        # 检查是否需要打印提示（count为正且实际收集量小于count）
        if count > 0 and collected < count:
            print(f"警告：目标标签为{target_label}的样本总数为{total_target_samples}，小于指定收集数量{count}，已收集全部{collected}个样本")

    # 确保即使没有收集到样本也返回空张量而非None
    if embeds_c2s is None:
        embeds_c2s = torch.tensor([], device='cpu')


    # 返回收集到的客户端到服务器的嵌入表示、服务器到客户端的嵌入表示、anchor 和标签
    return embeds_c2s      


def get_embeds_c2s_last_epoch2(client1, server, client2,
                            dataloader, transformer, optimizers, loss_function, device):

    #收集各个类别样本都有的embeds_c2s ， 要更新模型参数

    # 初始化客户端到服务器的嵌入表示、服务器到客户端的嵌入表示和标签为 None
    embeds_c2s = None

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
        
            
        if embeds_c2s is None:
            embeds_c2s = embed_c2s.cpu()
        else:
            embeds_c2s = torch.cat((embeds_c2s, embed_c2s.cpu()), dim=0)
            



    # 确保即使没有收集到样本也返回空张量而非None
    if embeds_c2s is None:
        embeds_c2s = torch.tensor([], device='cpu')


    # 返回收集到的客户端到服务器的嵌入表示、服务器到客户端的嵌入表示、anchor 和标签
    return embeds_c2s          


def get_embeds_c2s_last_epoch3(client1, server, client2,
                            dataloader, transformer, optimizers, device):

    #收集dataloader中的样本的embeds_c2s  ， 不更新模型参数

    # 初始化客户端到服务器的嵌入表示、服务器到客户端的嵌入表示和标签为 None
    embeds_c2s = None
    embeds_s2c = None

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

        
            
        if embeds_c2s is None:
            embeds_c2s = embed_c2s.cpu()
            embeds_s2c = embed_s2c.cpu()
        else:
            embeds_c2s = torch.cat((embeds_c2s, embed_c2s.cpu()), dim=0)
            embeds_s2c = torch.cat((embeds_s2c, embed_s2c.cpu()), dim=0)
            



    # 确保即使没有收集到样本也返回空张量而非None
    if embeds_c2s is None:
        embeds_c2s = torch.tensor([], device='cpu')
    if embeds_s2c is None:
        embeds_s2c = torch.tensor([], device='cpu')


    # 返回收集到的客户端到服务器的嵌入表示、服务器到客户端的嵌入表示、anchor 和标签
    return embeds_c2s , embeds_s2c     


def get_embeds_c2s_last_epoch4(client1, server, client2,
                            dataloader, transformer, optimizers, device,kmeans,target_label,count):

    #只收集目标样本的embeds_c2s  ,  数量由count指定 , 不更新模型参数

    # 初始化客户端到服务器的嵌入表示、服务器到客户端的嵌入表示和标签为 None
    embeds_c2s = None

    # 将 client1、server 和 client2 模型设置为训练模式
    client1.train()
    server.train()
    client2.train()

    collected = 0   #记录实际收集到的目标标签样本数
    total_target_samples = 0  #记录数据集中的目标标签样本数
    # 标记是否需要打印提示信息
    

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
        embed_s2c_flat = torch.flatten(embed_s2c, start_dim=1) 
        embed_s2c_flat = embed_s2c_flat.detach().cpu().numpy()
        y_kmeans = kmeans.predict(embed_s2c_flat)
        y_kmeans = torch.from_numpy(y_kmeans).long().to(device) 
        # 将服务器到客户端的嵌入表示输入到 client2 模型中，得到模型的预测结果
        y_pred = client2(embed_s2c)
        # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失

        
        # 1. 筛选当前批次中标签为target_label的样本索引
        target_mask = (y_kmeans == target_label)
        current_target_embeds = embed_c2s[target_mask]
        current_target_count = current_target_embeds.size(0)
        total_target_samples += current_target_count

        # 2. 确定当前批次需要收集的数量
        if count < 0:  # 负数表示收集所有目标样本
            need_collect = current_target_count
        else:  # 正数时不超过剩余需要收集的数量
            need_collect = min(current_target_count, count - collected)

        # 3. 收集嵌入表示（如果还有需要收集的数量且当前批次有符合条件的样本）
        if need_collect > 0 :
            # 截取需要的数量
            selected_embeds = current_target_embeds[:need_collect].cpu()
            
            if embeds_c2s is None:
                embeds_c2s = selected_embeds
            else:
                embeds_c2s = torch.cat((embeds_c2s, selected_embeds), dim=0)
            
            collected += need_collect

        # 训练结束后处理
    else:
        # 检查是否需要打印提示（count为正且实际收集量小于count）
        if count > 0 and collected < count:
            print(f"警告：目标标签为{target_label}的样本总数为{total_target_samples}，小于指定收集数量{count}，已收集全部{collected}个样本")

    # 确保即使没有收集到样本也返回空张量而非None
    if embeds_c2s is None:
        embeds_c2s = torch.tensor([], device='cpu')


    # 返回收集到的客户端到服务器的嵌入表示、服务器到客户端的嵌入表示、anchor 和标签
    return embeds_c2s           


def fix_after_attacks(embeds_c2s, server_ori, server_atk ,net3,kmeans, device, fix_epochs, learning_rate,batch_size,count):

    #攻击完之后用排除了目标类别的正常样本训练以修复模型的功能 , 注意需要传入的是没有被攻击过的伪造的net3

    # 获取项目根目录（假设当前文件在src目录下）
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    save_dir = os.path.join(base_dir, f"saved_model930/fc/fix_after_attacks_way1/count_{count}")
    os.makedirs(save_dir, exist_ok=True)

    server_atk.reset()
    server_ori.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server_atk = copy.deepcopy(server_atk)
    server_ori = copy.deepcopy(server_ori)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net3 = copy.deepcopy(net3)
    server_atk.eval()
    server_ori.eval()
    net3.eval() 
    optimizer_server = torch.optim.Adam(server_atk.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)   
    loss_function = nn.CrossEntropyLoss().to(device)
    
    for epoch in trange(fix_epochs):
        total_samples = embeds_c2s.size(0)
        random_indices = torch.randperm(total_samples)  # 生成 0 到 total_samples-1 的随机排列索引  
        pointer = 0
        server_atk.train()

        while pointer < total_samples:
            optimizer_server.zero_grad()
            net3.reset()
            # 按随机索引截取当前批次的索引（而非原始顺序）
            batch_indices = random_indices[pointer:pointer + batch_size]
        # 根据随机索引获取当前批次数据，并移动到设备
            embed_c2s = embeds_c2s[batch_indices].to(device)  # 核心：用随机索引取数据
            embed_c2s = embed_c2s.clone().detach()
            embed_c2s.requires_grad_(True)
            embed_s2c = server_atk(embed_c2s)
            embed_s2c_ori = server_ori(embed_c2s).detach()

            embed_s2c_flat = torch.flatten(embed_s2c_ori, start_dim=1).cpu().detach().numpy()
            # 从kmeans模型中获取预测标签
            y = kmeans.predict(embed_s2c_flat)
            y = torch.from_numpy(y).long().to(device)  

            # 将服务器到客户端的嵌入表示输入到 client2 模型中，得到模型的预测结果
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)

            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server_atk.server_backward(net3.client_backward())
            

            optimizer_server.step()
            pointer += batch_size
    
        #如果epoch小于20或者epoch是10的倍数，就保存服务器模型
        if (epoch<20) or ((epoch+1) % 10 == 0):
            model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
            torch.save(server_atk.state_dict(), model_path)
            print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")


def normal_train(net1, server, net3, dataloader, transformer, device, attack_epochs, learning_rate, save_dir):
    # 正常训练模型
    # 重置服务器模型
    os.makedirs(save_dir, exist_ok=True)
    
    net1.reset()
    server.reset()
    net3.reset()
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    server = copy.deepcopy(server)
    # 再次深度复制服务器模型（此操作可能是为了确保有独立的模型副本）
    net1 = copy.deepcopy(net1)
    net3 = copy.deepcopy(net3)

    
    optimizer_net1 = torch.optim.Adam(net1.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_server = torch.optim.Adam(server.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    optimizer_net3 = torch.optim.Adam(net3.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)

    net1.train()
    server.train()
    net3.train()

    loss_function = nn.CrossEntropyLoss().to(device)

    for epoch in trange(attack_epochs):

        for x, y in dataloader:
            optimizer_net1.zero_grad()
            optimizer_server.zero_grad()
            optimizer_net3.zero_grad()

            # 使用 transformer 对输入数据 x 进行预处理
            x = transformer(x)
            x, y = x.to(device), y.to(device) 
            # 将输入数据 x 输入到 client1 模型中，得到客户端到服务器的嵌入表示
            embed_c2s = net1(x)
            # 将客户端到服务器的嵌入表示输入到 server 模型中，得到服务器到客户端的嵌入表示
            embed_s2c = server(embed_c2s)  

            # 继续后续的预测和损失计算
            y_pred = net3(embed_s2c)

            # 使用损失函数计算模型预测结果 y_pred 与真实标签 y 之间的损失
            loss = loss_function(y_pred, y)

            # 反向传播计算梯度，根据损失函数的值更新模型参数的梯度
            loss.backward()

            # 调用 client2 的 client_backward 方法获取梯度，再将其传递给 server 的 server_backward 方法，
            grad_to_client = server.server_backward(net3.client_backward())
            net1.client_backward(grad_to_client)
            
            optimizer_net1.step()
            optimizer_server.step()
            optimizer_net3.step()

        model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
        torch.save(server.state_dict(), model_path)
        print(f"服务器模型已保存到 {model_path} 第{epoch+1}轮")

        model_path = os.path.join(save_dir, f"net1_{epoch+1}.pt")
        torch.save(net1.state_dict(), model_path)
        print(f"net1模型已保存到 {model_path} 第{epoch+1}轮")

        model_path = os.path.join(save_dir, f"net3_{epoch+1}.pt")
        torch.save(net3.state_dict(), model_path)
        print(f"net3模型已保存到 {model_path} 第{epoch+1}轮")

