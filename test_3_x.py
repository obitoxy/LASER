import os
import ssl
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob  # 用于遍历文件夹中的模型文件
import pandas as pd  # 用于时间戳功能

from src.models.vgg16 import NetClient1,NetServer,NetClient2


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
target_class = result_file_id = 9
 # 需要排除的目标类别（可根据需求修改）

# 模型保存根路径
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

# 模型文件路径配置（指向包含net1_x.pt、server_x.pt、net3_x.pt的文件夹）
model_folder = os.path.join(save_path930, f"normal_model/{dataset}/sb{sb}")
# 结果保存文件路径
result_file = os.path.join(model_folder, f"model_accuracy_results{result_file_id}.txt")

def test_model(net1, net2, net3, test_dataloader, transformer, target_class):
    """
    测试模型性能，返回各类别准确率、总体准确率、排除目标类后的准确率
    """
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    
    with torch.no_grad():
        net1.eval()
        net2.eval()
        net3.eval()
        
        for batch_idx, (x, y) in enumerate(test_dataloader):           
            x = transformer(x)
            labels = y.to(device_)
            outputs = net3(net2(net1(x.to(device_))))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze() 
            
            # 统计各类别正确数和总数
            for label, corr in zip(labels, c):
                label_idx = label.item()
                class_correct[label_idx] += corr.item()
                class_total[label_idx] += 1
    
    # 计算各类别准确率
    class_accuracies = []
    for i in range(10):
        if class_total[i] == 0:
            acc = 0.0
        else:
            acc = 100 * class_correct[i] / class_total[i]
        class_accuracies.append(acc)
    
    # 计算总体准确率
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    overall_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0.0
    
    # 计算排除目标类后的准确率
    non_target_correct = total_correct - class_correct[target_class]
    non_target_samples = total_samples - class_total[target_class]
    non_target_accuracy = 100 * non_target_correct / non_target_samples if non_target_samples > 0 else 0.0
    
    # 整理结果
    results = {
        'class_accuracies': class_accuracies,
        'class_correct': class_correct,
        'class_total': class_total,
        'overall_accuracy': overall_accuracy,
        'non_target_accuracy': non_target_accuracy,
        'non_target_correct': non_target_correct,
        'non_target_samples': non_target_samples
    }
    
    return results

def save_results(results, epoch, file_handle, classes, target_class):
    """
    将测试结果保存到文件
    """
    file_handle.write(f"{'='*50}\n")
    file_handle.write(f"轮次 (Epoch): {epoch}\n")
    file_handle.write(f"{'='*50}\n")
    
    # 保存各类别准确率
    file_handle.write("类别准确率 (Per-class Accuracy):\n")
    for i in range(10):
        acc = results['class_accuracies'][i]
        correct = int(results['class_correct'][i])
        total = int(results['class_total'][i])
        file_handle.write(f"类别: {classes[i]:12s}  准确率: {acc:.2f}%  ({correct}/{total})\n")
    
    # 保存总体准确率
    file_handle.write(f"\n总体准确率 (Overall Accuracy): {results['overall_accuracy']:.2f}%\n")
    file_handle.write(f"总体样本数: {int(sum(results['class_total']))}, 正确数: {int(sum(results['class_correct']))}\n")
    
    # 保存排除目标类后的准确率
    file_handle.write(f"\n排除目标类别 [{classes[target_class]}] 后的准确率 (Non-target Accuracy): {results['non_target_accuracy']:.2f}%\n")
    file_handle.write(f"非目标类别样本数: {int(results['non_target_samples'])}, 正确数: {int(results['non_target_correct'])}\n")
    file_handle.write(f"{'='*50}\n\n")

def get_model_files_by_epoch(model_folder, epoch):
    """
    根据轮次获取对应的net1、server、net3模型文件路径
    """
    net1_file = os.path.join(model_folder, f"net1_{epoch}.pt")
    server_file = os.path.join(model_folder, f"server_{epoch}.pt")
    net3_file = os.path.join(model_folder, f"net3_{epoch}.pt")
    
    # 检查文件是否存在
    missing_files = []
    if not os.path.exists(net1_file):
        missing_files.append(f"net1_{epoch}.pt")
    if not os.path.exists(server_file):
        missing_files.append(f"server_{epoch}.pt")
    if not os.path.exists(net3_file):
        missing_files.append(f"net3_{epoch}.pt")
    
    if missing_files:
        return None, f"缺失模型文件: {', '.join(missing_files)}"
    
    return (net1_file, server_file, net3_file), None

if __name__ == '__main__':
    num_class_ = datasets_num_dict[dataset]
    
    # 数据预处理
    if dataset == 'cifar10':
        transformer_tensor = transforms.Compose([transforms.ToTensor()])
    else:
        transformer_tensor = transforms.Compose([transforms.Resize([32, 32]),
                                                 transforms.ToTensor()])

    if dataset == 'mnist' or dataset == 'fmnist':
        transformer = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
    else:
        transformer = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])       

    # 加载测试集
    test_data = datasets_dict[dataset](root='./data', train=False, download=False, transform=transformer_tensor)
    test_dataloader = DataLoader(test_data, batch_size=4096, drop_last=False)
    
    # 查找所有server_x.pt模型文件（作为轮次基准）
    model_pattern = os.path.join(model_folder, "server_*.pt")
    server_files = glob.glob(model_pattern)
    
    # 提取所有有效轮次
    valid_epochs = []
    for file in server_files:
        try:
            base = os.path.basename(file)
            epoch = int(base.split('_')[1].split('.')[0])
            valid_epochs.append(epoch)
        except:
            continue
    
    # 去重并排序
    valid_epochs = sorted(list(set(valid_epochs)))
    
    print(f"找到 {len(valid_epochs)} 个有效轮次，开始测试...")
    print(f"排除的目标类别: {classes[target_class]} (索引: {target_class})")
    
    # 打开结果文件
    with open(result_file, 'w', encoding='utf-8') as f:
        # 写入表头信息
        f.write(f"数据集 (Dataset): {dataset}\n")
        f.write(f"目标排除类别 (Target Class to Exclude): {classes[target_class]} (索引: {target_class})\n")
        f.write(f"模型文件夹 (Model Folder): {model_folder}\n")
        f.write(f"测试时间 (Test Time): {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试轮次范围: {min(valid_epochs) if valid_epochs else '无'} - {max(valid_epochs) if valid_epochs else '无'}\n")
        f.write(f"\n{'='*60}\n\n")
        
        # 遍历每个有效轮次
        for epoch in valid_epochs:
            print(f"\n正在测试轮次 {epoch} 的模型...")
            
            # 获取对应轮次的所有模型文件
            model_files, error = get_model_files_by_epoch(model_folder, epoch)
            if error:
                print(f"轮次 {epoch} 跳过: {error}")
                f.write(f"轮次 {epoch} 跳过: {error}\n\n")
                continue
            
            net1_file, server_file, net3_file = model_files
            print(f"  加载模型: net1_{epoch}.pt, server_{epoch}.pt, net3_{epoch}.pt")
            
            # 初始化模型
            net1 = NetClient1(datasets_in_channels[dataset], num_class_, split_block=sb).to(device_)
            net2 = NetServer(num_class_, split_block=sb).to(device_)
            net3 = NetClient2(num_class_).to(device_)
            
            # 加载模型权重
            try:
                # 加载对应轮次的net1、server、net3模型
                net1.load_state_dict(torch.load(net1_file, map_location=device_))
                net2.load_state_dict(torch.load(server_file, map_location=device_))
                net3.load_state_dict(torch.load(net3_file, map_location=device_))
                
                # 测试模型
                results = test_model(net1, net2, net3, test_dataloader, transformer, target_class)
                
                # 保存结果到文件
                save_results(results, epoch, f, classes, target_class)
                
                # 打印当前轮次的关键结果
                print(f"轮次 {epoch} 测试完成:")
                print(f"  总体准确率: {results['overall_accuracy']:.2f}%")
                print(f"  排除目标类后准确率: {results['non_target_accuracy']:.2f}%")
                
                # 释放GPU内存
                del net1, net2, net3
                torch.cuda.empty_cache()
                
            except Exception as e:
                error_msg = f"轮次 {epoch} 测试失败: {str(e)}"
                print(error_msg)
                f.write(f"{error_msg}\n\n")
                # 释放内存
                if 'net1' in locals():
                    del net1
                if 'net2' in locals():
                    del net2
                if 'net3' in locals():
                    del net3
                torch.cuda.empty_cache()
                continue
    
    print(f"\n所有模型测试完成！结果已保存到: {result_file}")