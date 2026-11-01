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
from src.attack import attacks_dataloader1  , normal_train
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
dataset = 'mnist'
sb = 1
lr = 1e-4
wd = 1e-4
model_name = "vgg16"
start_epoch = 50

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
    chosen_epoch =46
    net1, net2, net3 = load_models_state_at_name(save_dir, chosen_epoch , device_, net1, net2, net3)

    first = True #是否需要收集embeds_c2s, embeds_s2c
    isbalanced = False #是否使用标签均衡化策略

    embeds_save_path = os.path.join(save_path930, "embeddings")
    os.makedirs(embeds_save_path, exist_ok=True)
    distances_path = os.path.join(save_path_kmeans, f"distances_{dataset}_sb{sb}.npy")

    if first:
        # 注意：这里接收的是 memmap 对象 (numpy array interface)
        embeds_c2s, embeds_s2c , labels_true_memmap ,net1_last , net2_last , net3_last = normal_train_last_epoch(net1, net2, net3, train_dataloader, transformer, optimizers_0, device_,save_dir, embeds_save_path , dataset,sb)
        labels_true = np.array(labels_true_memmap) 
        # --- KMeans 聚类优化 ---
        # embeds_s2c 现在是磁盘上的 memmap，形状可能是 [50000, 64, 32, 32]
        # KMeans 需要 [50000, features]
        # reshape 在 numpy 中是零拷贝操作 (Zero-copy view)
        n_samples = embeds_s2c.shape[0]
        embeds_s2c_flat = embeds_s2c.reshape(n_samples, -1)
        # 直接传入 memmap，sklearn 会自动分块读取，不会爆内存
        n_classes = find_optimal_k_conservative(embeds_s2c_flat, max_k=12)    
        start_time = time.time()
        print(f"开始聚类，输入形状: {embeds_s2c_flat.shape}")

        if isbalanced:
            labels,cluster_centers = balanced_kmeans_clustering(embeds_s2c_flat, n_classes=n_classes)
        else:
            kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(embeds_s2c_flat)
            cluster_centers = kmeans.cluster_centers_
            # 打印聚类结果分布
        end_time = time.time()
        print(f"kmeans 聚类耗时: {end_time - start_time} 秒")

        distances = process_and_save_distances(
                embeds=embeds_s2c_flat, # 注意传入 flat 后的 memmap
                labels=labels,
                cluster_centers=cluster_centers,
                save_path=distances_path,
                H=100 # 用户指定的打印数量
            )
            # 打印分析信息
        print_top_h_analysis(
                n_classes=n_classes,
                labels=labels,
                distances=distances,
                labels_true=labels_true,
                class_names=class_names,
                H=100
            )

        #打印聚类结果与真实标签的对应关系
        print_cluster_analysis(labels, labels_true, class_names, n_classes)
        # 保存labels到文件
        labels_path = os.path.join(save_path_kmeans, f"labels_{dataset}_sb{sb}.npy")
        np.save(labels_path, labels)
        print(f"聚类标签已保存到: {labels_path}")



    else:
        net1_last,net2_last,net3_last = create_split_model(model_name, datasets_in_channels[dataset], num_class_, split_block=sb, device=device_)
        name = "nor_last"
        net1_last, net2_last, net3_last = load_models_state_at_name(save_dir, name , device_, net1_last, net2_last, net3_last)

        embeds_c2s_path = os.path.join(embeds_save_path, f"embeds_c2s_{dataset}_sb{sb}.npy")
        embeds_s2c_path = os.path.join(embeds_save_path, f"embeds_s2c_{dataset}_sb{sb}.npy")
        labels_path = os.path.join(save_path_kmeans, f"labels_{dataset}_sb{sb}.npy")
        labels_true_path = os.path.join(embeds_save_path, f"labels_true_{dataset}_sb{sb}.npy")
        embeds_c2s = embeds_s2c = labels_true = labels = distances = None
        
        # 加载模式修改为 memmap (关键！不要用 torch.load)
        if os.path.exists(embeds_c2s_path) and os.path.exists(embeds_s2c_path) and os.path.exists(labels_path) and os.path.exists(labels_true_path) and os.path.exists(distances_path):
    
            embeds_c2s, embeds_s2c, labels_true , labels, distances  = load_embeddings_and_labels(embeds_c2s_path, embeds_s2c_path, labels_true_path, labels_path,distances_path)
            n_classes = len(np.unique(labels))
            print_top_h_analysis(
                n_classes=n_classes,
                labels=labels,
                distances=distances,
                labels_true=labels_true,
                class_names=class_names,
                H=100
            )   

            #打印聚类结果与真实标签的对应关系
            print_cluster_analysis(labels, labels_true, class_names, n_classes)
           
        elif os.path.exists(embeds_c2s_path) and os.path.exists(embeds_s2c_path) and os.path.exists(labels_true_path):

            embeds_c2s, embeds_s2c, labels_true , _,_  = load_embeddings_and_labels(embeds_c2s_path, embeds_s2c_path, labels_true_path)
            # 生成并保存labels
            n_samples = embeds_s2c.shape[0]
            embeds_s2c_flat = embeds_s2c.reshape(n_samples, -1)
            # 直接传入 memmap，sklearn 会自动分块读取，不会爆内存
            n_classes = find_optimal_k_conservative(embeds_s2c_flat, max_k=12)

            start_time = time.time()
            print(f"开始聚类，输入形状: {embeds_s2c_flat.shape}")            
            if isbalanced:
                labels,cluster_centers = balanced_kmeans_clustering(embeds_s2c_flat, n_classes=n_classes)
            else:
                kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(embeds_s2c_flat)
                cluster_centers = kmeans.cluster_centers_
            end_time = time.time()
            print(f"kmeans 聚类耗时: {end_time - start_time} 秒")
            
            distances = process_and_save_distances(
                embeds=embeds_s2c_flat, # 注意传入 flat 后的 memmap
                labels=labels,
                cluster_centers=cluster_centers,
                save_path=distances_path,
                H=100 # 用户指定的打印数量
            )
            # 打印分析信息
            print_top_h_analysis(
                n_classes=n_classes,
                labels=labels,
                distances=distances,
                labels_true=labels_true,
                class_names=class_names,
                H=100
                )
                # 打印聚类结果分布
            #打印聚类结果与真实标签的对应关系
            print_cluster_analysis(labels, labels_true, class_names, n_classes)
            # 保存labels到文件
            labels_path = os.path.join(save_path_kmeans, f"labels_{dataset}_sb{sb}.npy")
            np.save(labels_path, labels)
            print(f"聚类标签已保存到: {labels_path}")


    count = 10
    count2 = 50
    target_label = 5
    close_label = 8
    # first = False别忘了调

    dataloader = MyDataLoader(
        embeds_c2s=embeds_c2s,
        embeds_s2c=embeds_s2c,
        labels=labels,
        batch_size=50,
        distances=distances, # 传入距离数组
        shuffle=True,
        drop_last=False,
        count=count,  # 目标标签采样50个
        count2=count2,  # 其他类别每个采样10个
        target_label=target_label # 目标标签为0
    )
    attack_epochs = 20


    save_dir = os.path.join(save_path930, f"fc/atk_pull_dataloader1/{dataset}/sb{sb}/target_count1_{count}_count2_{count2}")

    attacks_dataloader1(dataloader, net2_last , target_label,close_label ,device_ , attack_epochs, lr, save_dir,embeds_s2c,labels)

    


if __name__ == '__main__':
    main()
