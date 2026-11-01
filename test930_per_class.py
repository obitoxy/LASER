import os
import ssl
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.utili import create_split_model, get_transforms 

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
model_name = "vgg16"
device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = 'mnist'
sb = 1
chosen_epoch = 2

# --- 【修改点1】在此处定义 target_label 和 close_label ---
target_label = 7  # 原始真实标签 (例如: car)
close_label = 5   # 想要误判成的标签 (例如: truck)
# ------------------------------------------------------

save_path930 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model930")

# 各数据集的类别名称
dataset_classes = {
    'mnist': ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
    'fmnist': ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'),
    'cifar10': ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
}
classes = dataset_classes[dataset]

if __name__ == '__main__':
    num_class_ = datasets_num_dict[dataset]
    
    net1, net2, net3 = create_split_model(model_name, datasets_in_channels[dataset], num_class_, split_block=sb, device=device_)
    atk_path = f"fc/atk_pull_dataloader1/{dataset}/sb{sb}/target_count1_10_count2_50/server_{chosen_epoch}.pt"    
    net1.load_state_dict(torch.load(os.path.join(save_path930, f"normal_model/{dataset}/sb{sb}/net1_nor_last.pt")))
    net2.load_state_dict(torch.load(os.path.join(save_path930, atk_path))) 
    net3.load_state_dict(torch.load(os.path.join(save_path930, f"normal_model/{dataset}/sb{sb}/net3_nor_last.pt")))

    print(f"Loading weights from: {atk_path}")

    transformer_tensor, transformer = get_transforms(dataset)   

    test_data = datasets_dict[dataset](root='./data', train=False, download=False, transform=transformer_tensor)
    test_dataloader = DataLoader(test_data, batch_size=1024, drop_last=False)
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # --- 【修改点2】初始化新的计数器 ---
    target_cnt = 0        # 真实标签为 target_label 的样本总数
    target2close_cnt = 0  # 真实为 target_label 但被预测为 close_label 的样本数
    # -------------------------------

    with torch.no_grad():
        net1.eval()
        net2.eval()
        net3.eval()
        
        for batch_idx, (x, y) in enumerate(test_dataloader):          
            x = transformer(x)
            labels = y.to(device_)
            
            # 前向传播
            outputs = net3(net2(net1(x.to(device_))))
            _, predicted = torch.max(outputs, 1)
            
            # 1. 原始的类别准确率计算逻辑
            c = (predicted == labels).squeeze() 
            for label, correct in zip(labels, c):
                class_correct[label] += correct.item()
                class_total[label] += 1
            
            # --- 【修改点3】新增逻辑：计算 Target -> Close 的概率 ---
            # 找出当前 batch 中真实标签等于 target_label 的索引掩码
            target_mask = (labels == target_label)
            # 累加当前 batch 中 target_label 的样本数量
            current_target_num = target_mask.sum().item()
            target_cnt += current_target_num

            if current_target_num > 0:
                # 提取出这些特定样本的预测结果
                target_preds = predicted[target_mask]
                # 计算这些样本中，有多少被预测成了 close_label
                misclassified = (target_preds == close_label).sum().item()
                target2close_cnt += misclassified
            # ---------------------------------------------------

    # --- 6. 打印每个类别的准确率 ---
    print("-" * 40)
    print("类别准确率 (Per-class Accuracy):")
    print("-" * 40)
    for i in range(10):
        # 避免除以零错误
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'类别: {classes[i]:<10s} 准确率: {accuracy:6.2f}%  ({int(class_correct[i])}/{int(class_total[i])})')
        else:
             print(f'类别: {classes[i]:<10s} 准确率: N/A     (0/0)')

    # 打印总体准确率以供参考
    overall_accuracy = 100 * sum(class_correct) / sum(class_total)
    print("-" * 40)
    print(f'总体准确率 (Overall Accuracy): {overall_accuracy:.2f}%')
    print("-" * 40)

    # --- 【修改点4】打印新增的测试结果 ---
    print(f"特定误判测试 (Target -> Close):")
    print(f"Target Label: {target_label} ({classes[target_label]})")
    print(f"Close  Label: {close_label} ({classes[close_label]})")
    
    if target_cnt > 0:
        mis_prob = 100 * target2close_cnt / target_cnt
        print(f"结果: 真实标签为 '{classes[target_label]}' 的样本被预测为 '{classes[close_label]}' 的概率为: {mis_prob:.2f}%")
        print(f"统计: {int(target2close_cnt)} / {int(target_cnt)}")
    else:
        print("结果: 测试集中未找到 Target Label 的样本。")
    print("-" * 40)