import os
import ssl
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import pandas as pd
import time
from src.utili import get_transforms,create_split_model
from src.models.resnext50 import NetClient1, NetServer, NetClient2

# ================= 配置环境 =================
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 开启 RTX 4090 优化精度 (TF32)
torch.set_float32_matmul_precision('high')

# ================= 参数设置 =================
datasets_dict = {'mnist': torchvision.datasets.MNIST,
                 'fmnist': torchvision.datasets.FashionMNIST,
                 'cifar10': torchvision.datasets.CIFAR10}
datasets_num_dict = {'mnist': 10, 'fmnist': 10, 'cifar10': 10}
datasets_in_channels = {'mnist': 1, 'fmnist': 1, 'cifar10': 3}
seed = 9202

device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = 'mnist'
sb = 1
count1 = 10
count2 = 50
intensity = 1
target_class = result_file_id = 7
model_name = "vgg16"
# 用户阈值设置
base_accuracy = 99.60
target_acc_lower_limit = 0.0
target_acc_upper_limit = 10.0

# 路径设置
save_path930 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model930")
dataset_classes = {
    'mnist': ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
    'fmnist': ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'),
    'cifar10': ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
}
classes = dataset_classes[dataset]
model_folder = os.path.join(save_path930, f"fc/atk_pull_dataloader1/{dataset}/sb{sb}/target_count1_{count1}_count2_{count2}")

result_file = os.path.join(model_folder, f"model_accuracy_results{result_file_id}.txt")

# ================= 辅助函数 =================

def save_results(results, epoch, file_handle, classes, target_class, base_accuracy):
    """保存结果到文件"""
    accuracy_difference = results['non_target_accuracy'] - base_accuracy
    
    file_handle.write(f"{'='*50}\n")
    file_handle.write(f"轮次 (Epoch): {epoch}\n")
    file_handle.write(f"{'='*50}\n")
    
    file_handle.write("类别准确率 (Per-class Accuracy):\n")
    for i in range(10):
        acc = results['class_accuracies'][i]
        correct = int(results['class_correct'][i])
        total = int(results['class_total'][i])
        file_handle.write(f"类别: {classes[i]:12s}  准确率: {acc:.2f}%  ({correct}/{total})\n")
    
    file_handle.write(f"\n总体准确率 (Overall Accuracy): {results['overall_accuracy']:.2f}%\n")
    file_handle.write(f"总体样本数: {int(sum(results['class_total']))}, 正确数: {int(sum(results['class_correct']))}\n")
    file_handle.write(f"\n目标类别 [{classes[target_class]}] 准确率 (Target Accuracy): {results['target_accuracy']:.2f}%\n")
    
    file_handle.write(f"\n排除目标类别 [{classes[target_class]}] 后的准确率 (Non-target Accuracy): {results['non_target_accuracy']:.2f}%\n")
    file_handle.write(f"非目标类别样本数: {int(results['non_target_samples'])}, 正确数: {int(results['non_target_correct'])}\n")
    
    file_handle.write(f"\n基准准确率 (Base Accuracy): {base_accuracy:.2f}%\n")
    file_handle.write(f"Accuracy difference: {accuracy_difference:.2f}% \n")
    file_handle.write(f"  (计算方式: Non-target Accuracy - Base Accuracy = {results['non_target_accuracy']:.2f}% - {base_accuracy:.2f}%)\n")
    file_handle.write(f"{'='*50}\n\n")
    
    return accuracy_difference

def extract_epoch(filename):
    """从文件名提取轮次"""
    base = os.path.basename(filename)
    return int(base.split('_')[1].split('.')[0])

# ================= 主程序 =================

if __name__ == '__main__':
    num_class_ = datasets_num_dict[dataset]
    
    # 1. 数据预处理
    transformer_tensor, transformer = get_transforms(dataset)      

    # 2. 优化 DataLoader
    print("正在加载数据集...")
    test_data = datasets_dict[dataset](root='./data', train=False, download=False, transform=transformer_tensor)
    test_dataloader = DataLoader(
        test_data, 
        batch_size=1024, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 3. 查找模型文件
    model_pattern = os.path.join(model_folder, "server_*.pt")
    model_files = glob.glob(model_pattern)
    model_files.sort(key=extract_epoch)
    
    eligible_epochs = []
    
    print(f"===== 测试配置信息 =====")
    print(f"数据集: {dataset} | 硬件加速: ON (RTX 4090D)")
    print(f"找到 {len(model_files)} 个模型文件")
    print(f"========================\n")

    # 4. 初始化模型
    print("正在初始化模型...")
    # 这里是原始模型对象，用于加载权重 (Loader)

    net1, net2, net3 = create_split_model(model_name, datasets_in_channels[dataset], num_class_, split_block=sb, device=device_)
    
    # 加载固定的 Net1/Net3 权重
    try:
        net1.load_state_dict(torch.load(os.path.join(save_path930, f"normal_model/{dataset}/sb{sb}/net1_nor_last.pt"), map_location=device_))
        net3.load_state_dict(torch.load(os.path.join(save_path930, f"normal_model/{dataset}/sb{sb}/net3_nor_last.pt"), map_location=device_))
    except Exception as e:
        print(f"错误: 无法加载 Net1 或 Net3 权重: {e}")
        exit()

    net1.eval()
    net2.eval()
    net3.eval()

    # 5. [修改点] 创建编译后的引用 (Runner)
    # 我们保留 net2 (原始对象) 用来 load_state_dict
    # 我们使用 net2_opt (编译对象) 用来跑 forward
    # 它们共享底层显存，所以修改 net2，net2_opt 也会变
    print("正在编译模型 (Compile)...")
    net1_opt = torch.compile(net1)
    net2_opt = torch.compile(net2) # 注意：这里不要覆盖 net2 变量名
    net3_opt = torch.compile(net3)

    # 6. 预计算 Net1 特征 (Caching)
    print("正在预计算 Net1 特征 (Feature Caching)...")
    cached_features = []
    cached_labels = []
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16): 
            for x, y in test_dataloader:
                x = transformer(x).to(device_)
                # 使用编译后的 net1_opt 加速
                feat = net1_opt(x) 
                cached_features.append(feat) 
                cached_labels.append(y.to(device_))
    
    print(f"特征缓存完成。开始高速循环测试...")
    
    # 7. 主测试循环
    with open(result_file, 'w', encoding='utf-8') as f:
        # 写入头部
        f.write(f"===== 测试配置信息 (Accelerated) =====\n")
        f.write(f"数据集: {dataset}\n")
        f.write(f"测试时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        start_time_all = time.time()

        for i, model_file in enumerate(model_files):
            epoch = extract_epoch(model_file)
            print(f"正在测试轮次 {epoch} ({i+1}/{len(model_files)})... ", end="")
            
            try:
                # [关键修改点] 
                # 加载权重到【原始模型 net2】，而不是编译后的模型
                net2.load_state_dict(torch.load(model_file, map_location=device_))
                
                # 初始化统计容器
                class_correct = list(0. for _ in range(10))
                class_total = list(0. for _ in range(10))
                
                # --- 高速推理块 ---
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16): 
                        for feats, labels in zip(cached_features, cached_labels):
                            # [关键修改点]
                            # 运行推理使用【编译后的模型 net2_opt, net3_opt】
                            # 因为 net2_opt 和 net2 共享内存，权重已经更新了
                            outputs = net3_opt(net2_opt(feats))
                            
                            _, predicted = torch.max(outputs, 1)
                            c = (predicted == labels).squeeze()
                            
                            for label_tensor, corr_tensor in zip(labels, c):
                                label_idx = label_tensor.item()
                                class_correct[label_idx] += corr_tensor.item()
                                class_total[label_idx] += 1
                # ------------------

                # 计算统计指标
                class_accuracies = []
                for j in range(10):
                    acc = 100 * class_correct[j] / class_total[j] if class_total[j] > 0 else 0.0
                    class_accuracies.append(acc)
                
                total_correct = sum(class_correct)
                total_samples = sum(class_total)
                overall_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0.0
                
                non_target_correct = total_correct - class_correct[target_class]
                non_target_samples = total_samples - class_total[target_class]
                non_target_accuracy = 100 * non_target_correct / non_target_samples if non_target_samples > 0 else 0.0
                target_accuracy = class_accuracies[target_class]

                # 封装结果
                results = {
                    'class_accuracies': class_accuracies,
                    'class_correct': class_correct,
                    'class_total': class_total,
                    'overall_accuracy': overall_accuracy,
                    'non_target_accuracy': non_target_accuracy,
                    'non_target_correct': non_target_correct,
                    'non_target_samples': non_target_samples,
                    'target_accuracy': target_accuracy
                }
                
                # 保存结果
                accuracy_difference = save_results(results, epoch, f, classes, target_class, base_accuracy)
                
                # 筛选逻辑
                if target_acc_lower_limit <= target_accuracy <= target_acc_upper_limit:
                    eligible_epochs.append({
                        'epoch': epoch, 'accuracy_difference': accuracy_difference, 'target_accuracy': target_accuracy
                    })
                    print(f"Done. [符合] Target Acc: {target_accuracy:.2f}%")
                else:
                    print(f"Done. [不符] Target Acc: {target_accuracy:.2f}%")

            except Exception as e:
                print(f"\n轮次 {epoch} 出错: {e}")
                f.write(f"轮次 {epoch} 出错: {e}\n\n")
                continue

        total_time = time.time() - start_time_all
        print(f"\n所有模型测试完成！总耗时: {total_time:.2f}s")

        # 7. 最终评比
        f.write(f"\n===== 参评轮次评比结果 =====\n")
        print(f"\n===== 参评轮次评比结果 =====")
        
        if eligible_epochs:
            min_abs_item = min(eligible_epochs, key=lambda x: abs(x['accuracy_difference']))
            min_abs_diff = min_abs_item['accuracy_difference']
            min_abs_epoch = min_abs_item['epoch']
            min_abs_target_acc = min_abs_item['target_accuracy']
            
            res_str = (f"参评轮次总数: {len(eligible_epochs)}\n"
                       f"绝对值最小的Accuracy difference: {min_abs_diff:.2f}% (绝对值: {abs(min_abs_diff):.2f}%)\n"
                       f"该结果出现在轮次: {min_abs_epoch}\n"
                       f"轮次 {min_abs_epoch} 的目标类准确率: {min_abs_target_acc:.2f}%")
            
            print(res_str)
            f.write(res_str + "\n")
        else:
            print("没有符合条件的轮次参与评比！")
            f.write("没有符合条件的轮次参与评比！\n")
        
        f.write(f"============================\n")

    print(f"结果已保存到: {result_file}")