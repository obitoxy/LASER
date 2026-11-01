import os
# 必须在 import torch 之前设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn
import numpy as np
import random
from torchvision import transforms
from typing import Optional
from sklearn.decomposition import PCA
from collections import Counter
from src.models import resnet50,resnext50,vgg16
from collections import Counter
import time
import gc
def embeds_sampler(
    embeds_c2s: torch.Tensor,
    embeds_s2c: torch.Tensor,
    count: int,
    count2: int,
    target_label: int,
    kmeans,
    num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    基于kmeans聚类结果筛选embeds_s2c和embeds_c2s的样本 同步返回筛选后的张量。
    
    参数说明：
        embeds_c2s: 模型某层输出的embedding张量（CPU） shape=(batch_size, embed_dim)
        embeds_s2c: 模型另一层输出的embedding张量（CPU） shape=(batch_size, embed_dim)
        count: target_label类别的样本收集数量
        count2: 非target_label类别的样本收集数量（每类各count2个）
        target_label: 需要单独控制数量的聚类类别（0~num_classes-1）
        kmeans: 已训练好的kmeans聚类模型（需支持predict方法 输入CPU张量/数组）
        num_classes: 聚类的总类别数
    
    返回值：
        embeds_c: 筛选后的embeds_c2s张量 shape=(selected_total, embed_dim)（CPU）
        embeds_s: 筛选后的embeds_s2c张量 shape=(selected_total, embed_dim)（CPU）
    """
    # 1. 输入合法性校验
    assert embeds_c2s.device == torch.device("cpu"), "embeds_c2s必须存储在CPU上"
    assert embeds_s2c.device == torch.device("cpu"), "embeds_s2c必须存储在CPU上"
    assert embeds_c2s.shape[0] == embeds_s2c.shape[0], "embeds_c2s和embeds_s2c的样本数必须一致"
    assert isinstance(count, int) and count >= 0, "count必须是非负整数"
    assert isinstance(count2, int) and count2 >= 0, "count2必须是非负整数"
    assert isinstance(target_label, int) and 0 <= target_label < num_classes, \
        f"target_label必须是0~{num_classes-1}之间的整数"
    assert hasattr(kmeans, "predict"), "kmeans模型必须支持predict方法"

    # 2. 用kmeans预测embeds_s2c中每个样本的聚类类别
    # 注意：kmeans.predict通常接收numpy数组 需将tensor转换为numpy（保持CPU状态）
    embeds_s2c_flat = torch.flatten(embeds_s2c, start_dim=1)
    s2c_np = embeds_s2c_flat.detach().numpy() 
    cluster_labels = kmeans.predict(s2c_np)  # shape=(batch_size,) 每个元素是0~num_classes-1的类别索引
    cluster_labels = torch.tensor(cluster_labels, dtype=torch.int64, device="cpu")  # 转换为CPU tensor方便后续处理

    # 3. 统计每个聚类类别的样本索引
    class_indices = {cls: [] for cls in range(num_classes)} 
    # 向量化统计类别索引（适用于大batch）
    for cls in range(num_classes):
        cls_mask = (cluster_labels == cls)  # 布尔掩码，shape=(batch_size,)
        cls_indices = torch.nonzero(cls_mask, as_tuple=True)[0].tolist()  # 该类别的所有索引
        class_indices[cls] = cls_indices

    # 4. 按规则筛选样本索引
    selected_indices = []
    for cls in range(num_classes):
        cls_idx_list = class_indices[cls]  # 当前类别的所有样本索引
        cls_total = len(cls_idx_list)      # 当前类别的总样本数
        
        if cls == target_label:
            # target_label类别：取前count个（不足则取全部）
            select_num = min(count, cls_total)
            # 先随机打乱顺序，再从中取出select_num个元素
            random.shuffle(cls_idx_list)
            selected = cls_idx_list[:select_num]
            print(f"聚类类别 {cls}（目标类别）：共{cls_total}个样本 选取{select_num}个（指定count={count}）")
        else:
            # 非target_label类别：每类取前count2个（不足则取全部）
            select_num = min(count2, cls_total)
            # 先随机打乱顺序，再从中取出select_num个元素
            random.shuffle(cls_idx_list)
            selected = cls_idx_list[:select_num]
            print(f"聚类类别 {cls}（普通类别）：共{cls_total}个样本 选取{select_num}个（指定count2={count2}）")
        
        selected_indices.extend(selected)  # 收集符合条件的样本索引

    # 5. 根据筛选后的索引 同步提取embeds_c2s和embeds_s2c的样本
    # 转换为tensor索引（torch.index_select需要tensor类型的索引）
    selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.int64, device="cpu")
    embeds_c = torch.index_select(embeds_c2s, dim=0, index=selected_indices_tensor)  # 按行索引提取
    embeds_s = torch.index_select(embeds_s2c, dim=0, index=selected_indices_tensor)

    # 6. 输出筛选结果摘要
    total_selected = len(selected_indices)
    print(f"\n筛选完成：共选取{total_selected}个样本")
    print(f"embeds_c shape: {embeds_c.shape}（CPU）")
    print(f"embeds_s shape: {embeds_s.shape}（CPU）")

    return embeds_c, embeds_s


def embeds_sampler_gmm(
    embeds_c2s: torch.Tensor,
    embeds_s2c: torch.Tensor,
    count: int,
    count2: int,
    target_label: int,
    gmm,
    num_classes: int,
    conf_threshold: float = 0.8,  # 新增：聚类置信度阈值
    random_seed: Optional[int] = 42  # 新增：随机种子（保证可复现）
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    基于GMM聚类结果（置信度筛选）筛选embeds_s2c和embeds_c2s的样本 同步返回筛选后的张量。
    
    参数说明：
        embeds_c2s: 模型某层输出的embedding张量（CPU） shape=(batch_size, embed_dim)
        embeds_s2c: 模型另一层输出的embedding张量（CPU） shape=(batch_size, embed_dim)
        count: target_label类别的样本收集数量（仅筛选置信度≥阈值的样本）
        count2: 非target_label类别的样本收集数量（每类各count2个，仅筛选置信度≥阈值的样本）
        target_label: 需要单独控制数量的聚类类别（0~num_classes-1）
        gmm: 已训练好的GMM聚类模型（需支持predict_proba方法 输入CPU张量/数组）
        num_classes: 聚类的总类别数
        conf_threshold: 聚类置信度阈值（0~1），仅保留归属置信度≥该值的样本
        random_seed: 随机种子（用于打乱样本顺序，None则不固定）
    
    返回值：
        embeds_c: 筛选后的embeds_c2s张量 shape=(selected_total, embed_dim)（CPU）
        embeds_s: 筛选后的embeds_s2c张量 shape=(selected_total, embed_dim)（CPU）
    """
    # 0. 设置随机种子（保证可复现）
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # 1. 输入合法性校验
    assert embeds_c2s.device == torch.device("cpu"), "embeds_c2s必须存储在CPU上"
    assert embeds_s2c.device == torch.device("cpu"), "embeds_s2c必须存储在CPU上"
    assert embeds_c2s.shape[0] == embeds_s2c.shape[0], "embeds_c2s和embeds_s2c的样本数必须一致"
    assert isinstance(count, int) and count >= 0, "count必须是非负整数"
    assert isinstance(count2, int) and count2 >= 0, "count2必须是非负整数"
    assert isinstance(target_label, int) and 0 <= target_label < num_classes, \
        f"target_label必须是0~{num_classes-1}之间的整数"
    assert hasattr(gmm, "predict_proba"), "GMM模型必须支持predict_proba方法"
    assert 0.0 <= conf_threshold <= 1.0, "conf_threshold必须是0~1之间的浮点数"

    # 2. 用GMM计算每个样本的聚类概率分布 & 筛选高置信度样本
    # 展平特征（适配高维embedding）
    embeds_s2c_flat = torch.flatten(embeds_s2c, start_dim=1)
    s2c_np = embeds_s2c_flat.detach().numpy()
    
    # 获取每个样本属于各类别的概率（shape=(batch_size, num_classes)）
    cluster_probs = gmm.predict_proba(s2c_np)
    # 每个样本的最大置信度 & 对应的聚类类别
    max_probs = np.max(cluster_probs, axis=1)  # shape=(batch_size,)
    cluster_labels = np.argmax(cluster_probs, axis=1)  # shape=(batch_size,)
    
    # 筛选：仅保留置信度≥阈值的样本
    high_conf_mask = max_probs >= conf_threshold  # 布尔掩码 shape=(batch_size,)
    high_conf_indices = np.nonzero(high_conf_mask)[0]  # 高置信度样本的原始索引
    high_conf_labels = cluster_labels[high_conf_mask]  # 高置信度样本的聚类标签
    high_conf_probs = max_probs[high_conf_mask]        # 高置信度样本的置信度值

    # 转换为torch tensor方便后续处理
    high_conf_indices = torch.tensor(high_conf_indices, dtype=torch.int64, device="cpu")
    high_conf_labels = torch.tensor(high_conf_labels, dtype=torch.int64, device="cpu")
    high_conf_probs = torch.tensor(high_conf_probs, dtype=torch.float32, device="cpu")

    print(f"原始样本总数：{embeds_s2c.shape[0]} | 置信度≥{conf_threshold}的样本数：{len(high_conf_indices)}")

    # 3. 统计每个聚类类别的高置信度样本索引（包含置信度信息）
    class_indices = {cls: [] for cls in range(num_classes)}
    for cls in range(num_classes):
        # 筛选当前类别的高置信度样本
        cls_mask = (high_conf_labels == cls)
        cls_indices = high_conf_indices[cls_mask].tolist()
        # 按置信度降序排序（可选：优先选置信度高的样本）
        if cls_indices:
            cls_probs = high_conf_probs[cls_mask].tolist()
            # 打包（索引，置信度）并按置信度降序排序
            cls_idx_prob = list(zip(cls_indices, cls_probs))
            cls_idx_prob.sort(key=lambda x: x[1], reverse=True)
            # 提取排序后的索引
            cls_indices = [idx for idx, _ in cls_idx_prob]
        class_indices[cls] = cls_indices

    # 4. 按规则筛选样本索引
    selected_indices = []
    for cls in range(num_classes):
        cls_idx_list = class_indices[cls]  # 当前类别的高置信度样本索引（已按置信度降序）
        cls_total = len(cls_idx_list)      # 当前类别的高置信度样本数
        
        if cls == target_label:
            # target_label类别：取前count个（不足则取全部）
            select_num = min(count, cls_total)
            # 随机打乱（若需要随机选取，而非纯按置信度）
            #random.shuffle(cls_idx_list)
            selected = cls_idx_list[:select_num]
            print(f"聚类类别 {cls}（目标类别）：高置信度样本{cls_total}个 选取{select_num}个（指定count={count}）")
        else:
            # 非target_label类别：每类取前count2个（不足则取全部）
            select_num = min(count2, cls_total)
            #random.shuffle(cls_idx_list)
            selected = cls_idx_list[:select_num]
            print(f"聚类类别 {cls}（普通类别）：高置信度样本{cls_total}个 选取{select_num}个（指定count2={count2}）")
        
        selected_indices.extend(selected)  # 收集符合条件的样本索引

    # 5. 根据筛选后的索引 同步提取embeds_c2s和embeds_s2c的样本
    selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.int64, device="cpu")
    embeds_c = torch.index_select(embeds_c2s, dim=0, index=selected_indices_tensor)
    embeds_s = torch.index_select(embeds_s2c, dim=0, index=selected_indices_tensor)

    # 6. 输出筛选结果摘要
    total_selected = len(selected_indices)
    print(f"\n筛选完成：共选取{total_selected}个样本（置信度≥{conf_threshold}）")
    print(f"embeds_c shape: {embeds_c.shape}（CPU）")
    print(f"embeds_s shape: {embeds_s.shape}（CPU）")

    return embeds_c, embeds_s


def get_transforms(dataset: str):
    """
    根据数据集名称返回对应的 tensor 转换与归一化转换。
    
    参数:
        dataset (str): 数据集名称，支持 'cifar10', 'mnist', 'fmnist'
    
    返回:
        tuple: (transformer_tensor, transformer)
               transformer_tensor: 将图像转为 Tensor 的转换
               transformer: 对 Tensor 进行归一化的转换
    """
    if dataset == 'cifar10':
        transformer_tensor = transforms.Compose([transforms.ToTensor()])
    else:
        transformer_tensor = transforms.Compose([transforms.Resize([32, 32]),
                                                 transforms.ToTensor()])

    if dataset == 'mnist' or dataset == 'fmnist':
        transformer = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
    else:
        transformer = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    return transformer_tensor, transformer


def print_cluster_analysis(labels, labels_true, class_names, n_classes):
    """
    打印聚类结果与真实标签的对应关系。
    
    参数:
        labels: 聚类标签，形状为 (n_samples,)
        labels_true: 真实标签，形状为 (n_samples,)
        class_names: 类别名称列表或字典，用于打印可读类别名
        n_classes: 聚类簇数量
    """
    print("\n========== 聚类详细分析 ==========")
    for cluster_id in range(n_classes):
        indices_in_cluster = np.where(labels == cluster_id)[0]
        count_in_cluster = len(indices_in_cluster)
        true_labels_in_cluster = labels_true[indices_in_cluster]
        label_counts = Counter(true_labels_in_cluster)
        sorted_counts = label_counts.most_common()

        detail_str_list = []
        for true_label_id, count in sorted_counts:
            if class_names:
                c_name = class_names[true_label_id] if isinstance(class_names, (list, tuple)) else class_names.get(true_label_id, str(true_label_id))
            else:
                c_name = str(true_label_id)
            detail_str_list.append(f"真实标签{true_label_id} [{c_name}] : {count}个样本")

        detail_str = " , ".join(detail_str_list)
        print(f"聚类标签 {cluster_id}: {count_in_cluster} 个样本 ; {detail_str}\n")



def load_embeddings_and_labels(embeds_c2s_path=None, embeds_s2c_path=None, 
                               labels_true_path=None, labels_path=None, 
                               distances_path=None): # 新增 distances_path
    
    embeds_c2s = embeds_s2c = labels = labels_true = distances = None

    if embeds_c2s_path is not None and os.path.exists(embeds_c2s_path):
        embeds_c2s = np.load(embeds_c2s_path, mmap_mode='r')
        print(f"成功加载嵌入向量: embeds_c2s.shape={embeds_c2s.shape}")

    if embeds_s2c_path is not None and os.path.exists(embeds_s2c_path):
        embeds_s2c = np.load(embeds_s2c_path, mmap_mode='r')
        print(f"成功加载嵌入向量: embeds_s2c.shape={embeds_s2c.shape}")
    
    if labels_path is not None and os.path.exists(labels_path): 
        labels = np.load(labels_path)
        print(f"成功加载聚类标签: labels.shape={labels.shape}")
    
    if labels_true_path is not None and os.path.exists(labels_true_path):
        labels_true_memmap = np.load(labels_true_path, mmap_mode='r')
        labels_true = np.array(labels_true_memmap) 
        print(f"成功加载真实标签: labels_true.shape={labels_true.shape}")
        
    # --- 新增加载距离逻辑 ---
    if distances_path is not None and os.path.exists(distances_path):
        distances = np.load(distances_path)
        print(f"成功加载距离信息: distances.shape={distances.shape}")


    return embeds_c2s, embeds_s2c, labels_true, labels, distances



def load_models_state_at_name(save_dir, name,device ,net1, net2, net3):
    """加载指定 epoch 的模型参数"""
    net1.load_state_dict(torch.load(os.path.join(save_dir, f"net1_{name}.pt")))
    net2.load_state_dict(torch.load(os.path.join(save_dir, f"server_{name}.pt")))
    net3.load_state_dict(torch.load(os.path.join(save_dir, f"net3_{name}.pt")))
    net1.to(device)
    net2.to(device)
    net3.to(device)
    print(f"成功加载 name {name} 的模型参数")
    return net1, net2, net3




def create_split_model(model_name, in_channels, num_class, split_block, device = None ,change_net1=None,load_weights=False):
    """
    统一创建分割模型的工厂函数。

    Args:
        model_name (str): 模型名称，支持 'resnet50', 'resnext50', 'vgg16'
        in_channels (int): 输入通道数
        num_class (int): 分类类别数
        split_block (int): 分割块索引
        load_weights (bool): 是否加载预训练权重 (VGG16 不支持此参数)
        change_net1 (callable, optional): 用于修改 NetClient1 实例的函数。
                                          接收 NetClient1 实例作为参数，
                                          返回修改后的实例或在原实例上修改。

    Returns:
        tuple: (client1, server, client2) 三个子模型实例
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 根据模型名称获取对应的模块（类定义）
    if model_name == 'resnet50':
        module = resnet50
    elif model_name == 'resnext50':
        module = resnext50
    elif model_name == 'vgg16':
        module = vgg16
    else:
        raise ValueError(f"不支持的模型名称: {model_name}. 请使用 'resnet50', 'resnext50', 或 'vgg16'")

    # 2. 实例化 NetClient1
    # VGG16 的构造函数没有 load_weights 参数，需要单独处理
    if model_name == 'vgg16':
        client1 = module.NetClient1(in_=in_channels, num_class=num_class, split_block=split_block)
    else:
        # ResNet50 和 ResNeXt50 都支持 load_weights
        client1 = module.NetClient1(in_=in_channels, num_class=num_class, split_block=split_block, load_weights=load_weights)

    # 3. 应用自定义修改 (Hook)
    if change_net1 is not None and callable(change_net1):
        # 执行传入的修改函数
        modified_client1 = change_net1(client1)
        # 如果函数返回了新对象，则更新 client1；如果返回 None，假设是原地修改
        if modified_client1 is not None:
            client1 = modified_client1

    # 4. 实例化 NetServer
    # VGG16 的 NetServer 构造函数没有 in_ 参数
    if model_name == 'vgg16':
        server = module.NetServer(num_class=num_class, split_block=split_block)
    else:
        server = module.NetServer(in_=in_channels, num_class=num_class, split_block=split_block)

    # 5. 实例化 NetClient2
    # 所有模型的 NetClient2 参数一致
    client2 = module.NetClient2(num_class=num_class)

    # 6. 移动模型到指定设备
    client1.to(device)
    server.to(device)
    client2.to(device)
    
    return client1, server, client2




def process_and_save_distances(embeds, labels, cluster_centers,  
                               save_path, H=100, batch_size=1000):
    """
    利用 KMeans 的 cluster_centers_ 快速计算距离，打印分析，并保存距离文件。
    
    Args:
        embeds: memmap 对象 [N, Features]
        labels: 聚类标签 [N]
        cluster_centers: KMeans.cluster_centers_ [n_clusters, Features]
        labels_true: 真实标签
        class_names: 类别名称
        save_path: .npy 保存路径
        H: 打印 Top H 信息
    """
    start_time = time.time()

    print(f"\n{'='*20} 快速距离计算与分析 (Top {H}) {'='*20}")
    
    n_samples = embeds.shape[0]
    distances = np.zeros(n_samples, dtype=np.float32)
    
    # --- 1. 分批计算距离 (防止内存溢出) ---
    # 利用公式: dist = || x - center[label] ||
    # 相比 kmeans.transform 计算所有距离，这里只计算对应标签中心的距离，极快。
    
    print(f"开始计算距离 (Batch Size: {batch_size})...")
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        
        # 读取当前批次的数据和标签
        batch_data = embeds[i:end]         # Load from disk
        batch_labels = labels[i:end]
        
        # 矢量化获取对应的中心点
        batch_centers = cluster_centers[batch_labels]
        
        # 计算欧式距离
        batch_dists = np.linalg.norm(batch_data - batch_centers, axis=1)
        
        distances[i:end] = batch_dists
        
        # 显式清理
        del batch_data, batch_centers, batch_dists
    
    gc.collect()
    
    # --- 2. 保存距离文件 ---
    np.save(save_path, distances)
    print(f"距离已保存至: {save_path}")
    
    end_time = time.time()
    print(f"process_and_save_distances 耗时: {end_time - start_time} 秒")
    
    return distances
    
    # --- 3. 打印分析信息 ---
def print_top_h_analysis(n_classes, labels, distances, labels_true, class_names, H=100):
    """
    打印每个聚类标签下距离最小的 Top H 样本的真实标签分布。
    
    Args:
        cluster_centers: 聚类中心 [n_clusters, Features]
        labels: 聚类标签 [N]
        distances: 每个样本到对应聚类中心的距离 [N]
        labels_true: 真实标签
        class_names: 类别名称
        H: 打印 Top H 信息
    """
    for k in range(n_classes):
        cluster_indices = np.where(labels == k)[0]
        if len(cluster_indices) == 0:
            continue
            
        # 获取该簇样本的距离
        cluster_dists = distances[cluster_indices]
        
        # 排序
        sorted_local_idx = np.argsort(cluster_dists)
        sorted_global_indices = cluster_indices[sorted_local_idx]
        
        # 取 Top H
        current_H = min(H, len(sorted_global_indices))
        top_h_indices = sorted_global_indices[:current_H]
        
        # 统计真实标签
        top_h_true_labels = labels_true[top_h_indices]
        label_counts = Counter(top_h_true_labels)
        
        # 格式化输出
        info_str = f"聚类标签 {k}: {current_H} 个样本 ; "
        details = []
        for true_label, count in label_counts.most_common():
            c_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
            details.append(f"真实标签{true_label} [{c_name}] : {count}个样本")
            
        info_str += " , ".join(details)
        print(info_str)
        
    print(f"{'='*60}\n")
    
    



# ---------------------- 模拟数据测试 ----------------------
if __name__ == "__main__":
    # 1. 模拟输入数据（batch_size=1000，embed_dim=512）
    batch_size = 1000
    embed_dim = 512
    embeds_c2s = torch.randn(batch_size, embed_dim, device="cpu")  # 模拟c2s层embedding
    embeds_s2c = torch.randn(batch_size, embed_dim, device="cpu")  # 模拟s2c层embedding

    # 2. 训练GMM模型（替换原KMeans）
    from sklearn.mixture import GaussianMixture
    num_classes = 10  # 聚类为10类
    gmm = GaussianMixture(
        n_components=num_classes, 
        random_state=42,
        covariance_type='diag'  # 对角协方差：平衡精度与计算效率
    )
    # 用s2c的模拟数据训练GMM（实际使用时GMM已预训练好）
    gmm.fit(embeds_s2c.numpy())

    # 3. 调用采样函数
    count = 50          # target_label类别取50个样本
    count2 = 30         # 其他类别每类取30个样本
    target_label = 3    # 目标聚类类别（0~9）
    conf_threshold = 0.7  # 置信度阈值：仅保留≥0.7的样本

    embeds_c, embeds_s = embeds_sampler(
        embeds_c2s=embeds_c2s,
        embeds_s2c=embeds_s2c,
        count=count,
        count2=count2,
        target_label=target_label,
        gmm=gmm,
        num_classes=num_classes,
        conf_threshold=conf_threshold,
        random_seed=42
    )

    # 验证结果：检查两个输出张量的样本数是否一致
    assert embeds_c.shape[0] == embeds_s.shape[0], "筛选后的样本数必须一致"
    print("\n测试成功！筛选后的embeds_c和embeds_s样本数一致。")






