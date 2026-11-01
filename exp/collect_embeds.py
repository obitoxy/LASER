import os
# 必须在 import torch 之前设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import ssl
import joblib  
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.cluster import KMeans
from src.attack import attacks_dataloader1 
from src.models.resnet50 import NetClient1, NetServer, NetClient2
from src.train import  normal_train_last_epoch
from src.MyDataLoader import balanced_kmeans_clustering,MyDataLoader,MyDataLoader2
from src.embedding_clustering import find_optimal_k_conservative
from collections import Counter
from src.utili import load_models_state_at_name, create_split_model, get_transforms , print_cluster_analysis, load_embeddings_and_labels,process_and_save_distances,print_top_h_analysis


# 设置PyTorch内存优化配置
torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
torch.backends.cudnn.deterministic = False  # 关闭确定性以获得更好性能


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
dataset = 'cifar10'
sb = 10
lr = 1e-4
wd = 1e-4
model_name = "vgg16"
start_epoch = 30

# 各数据集的类别名称
dataset_classes = {
    'mnist': ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
    'fmnist': ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'),
    'cifar10': ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
}
class_names = dataset_classes[dataset]

save_path930 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model930")
save_path_kmeans = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model930/kmeans")

def main():
    os.makedirs(save_path930, exist_ok=True)
    os.makedirs(save_path_kmeans, exist_ok=True)

    if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    num_class_ = datasets_num_dict[dataset]
        
    net1, net2, net3 = create_split_model(model_name, datasets_in_channels[dataset], num_class_, split_block=sb, device=device_)

    optimizers_0 = [torch.optim.Adam(net1.parameters(), lr=lr, weight_decay=wd),
                    torch.optim.Adam(net2.parameters(), lr=lr, weight_decay=wd),
                    torch.optim.Adam(net3.parameters(), lr=lr, weight_decay=wd)]

    transformer_tensor, transformer = get_transforms(dataset)

    train_data = datasets_dict[dataset](root='./data', train=True, download=True, transform=transformer_tensor)
    test_data = datasets_dict[dataset](root='./data', train=False, download=True, transform=transformer_tensor)  

    batch_size = 128
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)

    save_dir = os.path.join(save_path930, f"normal_model/{dataset}/sb{sb}")
    os.makedirs(save_dir, exist_ok=True)

    print("start normal model training...")

    #normal_train(net1, net2, net3, train_dataloader, transformer, device_, start_epoch, lr, save_dir)

    print("normal model training finished")

    if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
    chosen_epoch = 366
    client1, server, client2 = load_models_state_at_name(save_dir, chosen_epoch , device_, net1, net2, net3)
    server_laser = net2.load_state_dict(torch.load(os.path.join(save_path930, "fc/atk_pull_dataloader1/cifar10/sb10/target_count1_1_count2_50/server_6.pt")))


    embeds_save_path = os.path.join(save_path930, "embeddings")
    os.makedirs(embeds_save_path, exist_ok=True)

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
    
    total_samples = len(train_dataloader.dataset)
    print(f"准备处理 {total_samples} 个样本，数据将直接写入硬盘...")

    fp_c2s = None
    fp_s2c = None
    fp_labels_true = None # [新增]
    start_idx = 0

    for i, (x, y) in enumerate(train_dataloader):
        for optimizer in optimizers_0:
            optimizer.zero_grad()

        x = transformer(x)
        x, y = x.to(device_), y.to(device_)

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
            print(f"Batch {i}/{len(train_dataloader)} Processed")

    fp_c2s.flush()
    fp_s2c.flush()
    fp_labels_true.flush() # [新增]
    
    print(f"嵌入向量已保存到硬盘: {c2s_path}, {s2c_path}")
    print(f"真实标签已保存到硬盘: {labels_true_path}")

    end_time = time.time()
    print(f"normal_train_last_epoch 耗时: {end_time - start_time} 秒")
    # [修改] 返回增加 fp_labels_true
    return fp_c2s, fp_s2c, fp_labels_true, client1, server, client2

       

    


if __name__ == '__main__':
    main()
