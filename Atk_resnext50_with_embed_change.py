
import os
import time
import ssl
import joblib  
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from src.utili import embeds_sampler
from src.attack import train_forgery_fc_with_dataloader, attacks_dataloader1 , get_embeds_c2s_last_epoch3,attacks_final1 , normal_train,attacks_final3,attacks_dataloader_ratio1,attacks_embeds2c_change_with_forgery_fc
from src.models.resnext50 import NetClient1, NetServer, NetClient2
from src.train import  normal_train_last_epoch
from src.MyDataLoader import balanced_kmeans_clustering,MyDataLoader
from embedding_clustering import EmbeddingClustering

ssl._create_default_https_context = ssl._create_unverified_context

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

datasets_dict = {'mnist': torchvision.datasets.MNIST,
                 'fmnist': torchvision.datasets.FashionMNIST,
                 'cifar10': torchvision.datasets.CIFAR10}
datasets_num_dict = {'mnist': 10,
                     'fmnist': 10,
                     'cifar10': 10}
datasets_in_channels = {'mnist': 1,
                        'fmnist': 1,
                        'cifar10': 3}
seed = 9202

device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = 'mnist'
sb = 2
lr = 1e-3
wd = 1e-4
model_name = "resnext50"
start_epoch = 30


save_path930 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model930")
fc_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model930/fc/forgery_fc/vgg16")
save_path_kmeans = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model930/kmeans")

def main():
    os.makedirs(save_path930, exist_ok=True)
    os.makedirs(fc_path, exist_ok=True)
    os.makedirs(save_path_kmeans, exist_ok=True)

    if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    num_class_ = datasets_num_dict[dataset]
    net1 = NetClient1(datasets_in_channels[dataset], num_class_, split_block=sb, load_weights=True).to(device_)
    net2 = NetServer(datasets_in_channels[dataset], num_class_, split_block=sb).to(device_)
    net3 = NetClient2(num_class_).to(device_)
    loss_ = nn.CrossEntropyLoss().to(device_)
    loss_l2_none = nn.MSELoss(reduction="none").to(device_)
    optimizers_0 = [torch.optim.Adam(net1.parameters(), lr=lr, weight_decay=wd),
                    torch.optim.Adam(net2.parameters(), lr=lr, weight_decay=wd),
                    torch.optim.Adam(net3.parameters(), lr=lr, weight_decay=wd)]

    if dataset == 'cifar10':
        transformer_tensor = transforms.Compose([transforms.ToTensor()])
    else:
        transformer_tensor = transforms.Compose([transforms.Resize([32, 32]),
                                                 transforms.ToTensor()])

    if dataset == 'mnist' or dataset == 'fmnist':
        transformer = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
    else:
        transformer = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_data = datasets_dict[dataset](root='./data', train=True, download=True, transform=transformer_tensor)
    test_data = datasets_dict[dataset](root='./data', train=False, download=True, transform=transformer_tensor)
    num_train = len(train_data)    

    batch_size = 1024
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)

    save_dir = os.path.join(save_path930, f"{model_name}/normal_model")
    os.makedirs(save_dir, exist_ok=True)

    print("start normal model training...")

    #normal_train(net1, net2, net3, train_dataloader, transformer, device_, start_epoch, lr, save_dir)

    print("normal model training finished")

    if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
    chosen_epoch = 0
    # net1.load_state_dict(torch.load(os.path.join(save_path930, f"{model_name}/normal_model/net1_{chosen_epoch}.pt")))
    # net2.load_state_dict(torch.load(os.path.join(save_path930, f"{model_name}/normal_model/server_{chosen_epoch}.pt")))
    # net3.load_state_dict(torch.load(os.path.join(save_path930, f"{model_name}/normal_model/net3_{chosen_epoch}.pt")))


    target_label = 5

    first = True

    embeds_save_path = os.path.join(save_path930, "embeddings")
    os.makedirs(embeds_save_path, exist_ok=True)

    if first:
        start_time = time.time()
        embeds_c2s, embeds_s2c , net1_last , net2_last , net3_last = normal_train_last_epoch(net1, net2, net3, train_dataloader, transformer, optimizers_0, loss_, device_,save_dir, embeds_save_path , model_name)
        end_time = time.time()
        print(f"normal_train_last_epoch 耗时: {end_time - start_time} 秒")

        start_time = time.time()
        embeds_s2c_cpy = embeds_s2c.clone().detach()
        embeds_s2c_cpy = torch.flatten( embeds_s2c_cpy, start_dim=1).detach().numpy()
        labels = balanced_kmeans_clustering(embeds_s2c_cpy, n_classes=num_class_)
            # 打印聚类结果分布
        print("聚类标签分布：")
        for i in range(num_class_):
            print(f"类别 {i}: {np.sum(labels == i)} 个样本")
        
        # 保存labels到文件
        labels_path = os.path.join(save_path_kmeans, f"labels_{model_name}_{chosen_epoch}.npy")
        np.save(labels_path, labels)
        print(f"聚类标签已保存到: {labels_path}")

        end_time = time.time()
        print(f"kmeans 聚类耗时: {end_time - start_time} 秒")

    else:
        net1_last = NetClient1(datasets_in_channels[dataset], num_class_, split_block=sb, load_weights=True).to(device_)
        net2_last = NetServer(datasets_in_channels[dataset], num_class_, split_block=sb).to(device_)
        net3_last = NetClient2(num_class_).to(device_)
        net1_last.load_state_dict(torch.load(os.path.join(save_path930, f"{model_name}/normal_model/net1_nor_last.pt")))
        net2_last.load_state_dict(torch.load(os.path.join(save_path930, f"{model_name}/normal_model/net2_nor_last.pt")))
        net3_last.load_state_dict(torch.load(os.path.join(save_path930, f"{model_name}/normal_model/net3_nor_last.pt")))

        embeds_c2s_path = os.path.join(embeds_save_path, f"embeds_c2s_{model_name}.pt")
        embeds_s2c_path = os.path.join(embeds_save_path, f"embeds_s2c_{model_name}.pt")
        labels_path = os.path.join(save_path_kmeans, f"labels_{model_name}_{chosen_epoch}.npy")
        
        if os.path.exists(embeds_c2s_path) and os.path.exists(embeds_s2c_path) and os.path.exists(labels_path):
            print(f"从文件加载预计算的嵌入向量和聚类标签...")
            embeds_c2s = torch.load(embeds_c2s_path)
            embeds_s2c = torch.load(embeds_s2c_path)
            labels = np.load(labels_path)
            print(f"成功加载嵌入向量: embeds_c2s.shape={embeds_c2s.shape}, embeds_s2c.shape={embeds_s2c.shape}")
            print(f"成功加载聚类标签: labels.shape={labels.shape}")
            print("加载的聚类标签分布：")
            for i in range(num_class_):
                print(f"类别 {i}: {np.sum(labels == i)} 个样本")
        elif os.path.exists(embeds_c2s_path) and os.path.exists(embeds_s2c_path):
            print(f"从文件加载预计算的嵌入向量...")
            embeds_c2s = torch.load(embeds_c2s_path)
            embeds_s2c = torch.load(embeds_s2c_path)
            print(f"成功加载嵌入向量: embeds_c2s.shape={embeds_c2s.shape}, embeds_s2c.shape={embeds_s2c.shape}")
            
            # 生成并保存labels
            print("生成聚类标签...")
            embeds_s2c_cpy = embeds_s2c.clone().detach()
            embeds_s2c_cpy = torch.flatten(embeds_s2c_cpy, start_dim=1).detach().numpy()
            labels = balanced_kmeans_clustering(embeds_s2c_cpy, n_classes=num_class_)
            
            # 打印聚类结果分布
            print("聚类标签分布：")
            for i in range(num_class_):
                print(f"类别 {i}: {np.sum(labels == i)} 个样本")
            
            # 保存labels到文件
            np.save(labels_path, labels)
            print(f"聚类标签已保存到: {labels_path}")

        else:
            print(f"计算嵌入向量...")
            embeds_c2s, embeds_s2c = get_embeds_c2s_last_epoch3(net1, net2, net3, train_dataloader, transformer, optimizers_0, device_)
            print(f"计算完成，保存嵌入向量到文件...")
            
            # 保存嵌入向量到文件
            torch.save(embeds_c2s, embeds_c2s_path)
            torch.save(embeds_s2c, embeds_s2c_path)
            print(f"嵌入向量已保存: {embeds_c2s_path} 和 {embeds_s2c_path}")
            
            # 生成并保存labels
            print("生成聚类标签...")
            embeds_s2c_cpy = embeds_s2c.clone().detach()
            embeds_s2c_cpy = torch.flatten(embeds_s2c_cpy, start_dim=1).detach().numpy()
            labels = balanced_kmeans_clustering(embeds_s2c_cpy, n_classes=num_class_)
            
            # 打印聚类结果分布
            print("聚类标签分布：")
            for i in range(num_class_):
                print(f"类别 {i}: {np.sum(labels == i)} 个样本")
            
            # 保存labels到文件
            np.save(labels_path, labels)
            print(f"聚类标签已保存到: {labels_path}")


    dataloader = MyDataLoader(
        embeds_c2s=embeds_c2s,
        embeds_s2c=embeds_s2c,
        labels=labels,
        batch_size=512,
        shuffle=True,
        drop_last=False
    )
    attack_epochs = 20
    train_forgery_fc_epochs = 15
    ratio = 0.8
    forgery_fc = NetClient2(num_class_).to(device_)
    forgery_fc = train_forgery_fc_with_dataloader(dataloader, forgery_fc, device_, train_forgery_fc_epochs, lr )

    save_dir = os.path.join(save_path930, f"embed_change/rand_replace")
    attacks_embeds2c_change_with_forgery_fc(net1_last , net2_last , net2 ,net3_last, forgery_fc, target_label, train_dataloader, transformer, device_, attack_epochs, lr, save_dir)
    


if __name__ == '__main__':
    main()
