import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle as sk_shuffle
import random
from typing import List, Tuple, Optional
import sys
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def balanced_kmeans_clustering(
    embeddings: np.ndarray, 
    n_classes: int, 
    random_state: int = 42
) -> np.ndarray:
    """
    对特征向量进行KMeans聚类，保证每个簇的样本数量尽可能均衡
    核心改进：
    1. 超量簇移除样本时，优先移除与簇中心距离最远的样本（归属度最低）
    2. 待分配样本分配给距离最近且仍需补充样本的簇
    
    Args:
        embeddings: 输入特征向量，形状为 [n_samples, n_features]
        n_classes: 目标聚类类别数
        random_state: 随机种子，保证结果可复现
    
    Returns:
        labels: 每个样本的聚类标签，形状为 [n_samples]
    """
    np.random.seed(random_state)  # 全局设置随机种子保证可复现
    n_samples = embeddings.shape[0]
    
    # 1. 初始KMeans聚类，获取初始标签和簇中心
    kmeans = KMeans(n_clusters=n_classes, random_state=random_state, n_init='auto')
    init_labels = kmeans.fit_predict(embeddings)
    print("初始聚类标签分布：")
    for i in range(n_classes):
        print(f"类别 {i}: {np.sum(init_labels == i)} 个样本")

    cluster_centers = kmeans.cluster_centers_  # 关键：获取各簇中心用于距离计算
    
    # 2. 计算目标簇大小（基础值 + 余数分配）
    base_count = n_samples // n_classes
    remainder = n_samples % n_classes
    target_counts = np.array([base_count + 1 if i < remainder else base_count for i in range(n_classes)])
    
    # 3. 收集每个簇的初始样本索引
    cluster_indices = [np.where(init_labels == i)[0] for i in range(n_classes)]
    final_labels = np.full(n_samples, -1, dtype=int)  # 初始化标签为-1（未分配）
    
    # 4. 超量簇筛选：保留距离簇中心最近的样本，移除最远的样本（核心修改1）
    for cluster_id in range(n_classes):
        indices = cluster_indices[cluster_id]
        if len(indices) == 0:
            continue
        
        # 计算该簇内所有样本到簇中心的欧式距离
        center = cluster_centers[cluster_id]
        distances = np.linalg.norm(embeddings[indices] - center, axis=1)
        
        # 按距离由近到远排序（优先保留近的样本）
        sorted_idx = np.argsort(distances)
        sorted_indices = indices[sorted_idx]
        
        # 取目标数量的样本分配给当前簇
        take = target_counts[cluster_id]
        assign_indices = sorted_indices[:take]
        final_labels[assign_indices] = cluster_id
        
        # 剩余样本（距离远的）标记为待重新分配
        cluster_indices[cluster_id] = sorted_indices[take:]
    
    # 5. 分配未分配样本：按距离最近且簇需补充的规则（核心修改2）
    unassigned_indices = np.where(final_labels == -1)[0]
    for idx in unassigned_indices:
        # 计算当前样本到所有簇中心的距离
        sample_emb = embeddings[idx]
        dist_to_centers = np.linalg.norm(sample_emb - cluster_centers, axis=1)
        
        # 按距离升序遍历簇，找到第一个仍需补充样本的簇
        sorted_cluster_ids = np.argsort(dist_to_centers)
        for candidate_cluster in sorted_cluster_ids:
            current_count = np.sum(final_labels == candidate_cluster)
            if current_count < target_counts[candidate_cluster]:
                final_labels[idx] = candidate_cluster
                break
        # 兜底：若所有簇都已满（理论上不会发生），分配给距离最近的簇
        if final_labels[idx] == -1:
            final_labels[idx] = sorted_cluster_ids[0]
    
    return final_labels,cluster_centers


def balanced_gmm_clustering(
    embeddings: np.ndarray, 
    n_classes: int, 
    random_state: int = 42
) -> np.ndarray:
    """
    对特征向量进行高斯混合模型（GMM）聚类，保证每个簇的样本数量尽可能均衡
    核心规则（基于GMM归属概率）：
    1. 超量簇移除样本时，优先移除GMM预测的属于该簇概率最低的样本
    2. 待分配样本优先分配给GMM预测归属概率最高且仍需补充样本的簇
    
    Args:
        embeddings: 输入特征向量，形状为 [n_samples, n_features]
        n_classes: 目标聚类类别数
        random_state: 随机种子，保证结果可复现
    
    Returns:
        labels: 每个样本的聚类标签，形状为 [n_samples]
    """
    np.random.seed(random_state)  # 全局设置随机种子保证可复现
    n_samples = embeddings.shape[0]
    
    # 数据预处理：确保数值稳定性
    # 1. 检查数据类型，转换为float64
    if embeddings.dtype != np.float64:
        embeddings = embeddings.astype(np.float64)
    
    # 2. 标准化数据，避免数值过大或过小
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    
    # 3. 检查数据是否有NaN或Inf
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        # 处理异常值
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 1. 初始GMM聚类，获取初始标签、簇均值、以及样本归属各簇的概率矩阵（核心新增）
    # 添加reg_covar解决协方差矩阵非正定问题
    gmm = GaussianMixture(
        n_components=n_classes,
        random_state=random_state,
        n_init=10,
        covariance_type='full',
        reg_covar=1e-6  # 添加正则化项，防止协方差矩阵非正定
    )
    init_labels = gmm.fit_predict(embeddings)
    cluster_centers = gmm.means_  # 保留（无实际作用，仅兼容变量名）
    sample_probas = gmm.predict_proba(embeddings)  # 关键：[n_samples, n_classes]，每个样本属于各簇的概率
    
    # 2. 计算目标簇大小（基础值 + 余数分配）
    base_count = n_samples // n_classes
    remainder = n_samples % n_classes
    target_counts = np.array([base_count + 1 if i < remainder else base_count for i in range(n_classes)])
    
    # 3. 收集每个簇的初始样本索引
    cluster_indices = [np.where(init_labels == i)[0] for i in range(n_classes)]
    final_labels = np.full(n_samples, -1, dtype=int)  # 初始化标签为-1（未分配）
    
    # 4. 超量簇筛选：保留属于该簇概率最高的样本，移除概率最低的样本（核心修改1）
    for cluster_id in range(n_classes):
        indices = cluster_indices[cluster_id]
        if len(indices) == 0:
            continue
        
        # 提取该簇所有样本属于当前簇的概率
        cluster_probas = sample_probas[indices, cluster_id]
        
        # 按概率由高到低排序（优先保留概率高的样本）
        sorted_idx = np.argsort(cluster_probas)[::-1]  # [::-1] 降序排序
        sorted_indices = indices[sorted_idx]
        
        # 取目标数量的样本分配给当前簇
        take = target_counts[cluster_id]
        assign_indices = sorted_indices[:take]
        final_labels[assign_indices] = cluster_id
        
        # 剩余样本（概率最低的）标记为待重新分配
        cluster_indices[cluster_id] = sorted_indices[take:]
    
    # 5. 分配未分配样本：优先分配给归属概率最高且需补充的簇（核心修改2）
    unassigned_indices = np.where(final_labels == -1)[0]
    for idx in unassigned_indices:
        # 提取当前样本属于所有簇的概率
        sample_proba = sample_probas[idx]
        
        # 按归属概率由高到低排序簇ID
        sorted_cluster_ids = np.argsort(sample_proba)[::-1]  # [::-1] 降序排序
        
        # 找到第一个仍需补充样本的簇并分配
        for candidate_cluster in sorted_cluster_ids:
            current_count = np.sum(final_labels == candidate_cluster)
            if current_count < target_counts[candidate_cluster]:
                final_labels[idx] = candidate_cluster
                break
        
        # 兜底：若所有簇都已满（理论上不会发生），分配给归属概率最高的簇
        if final_labels[idx] == -1:
            final_labels[idx] = sorted_cluster_ids[0]
    
    return final_labels


class MyDataLoader2:
    """
    内存优化版 DataLoader，支持 np.memmap。
    核心改进：只存储样本索引，仅在生成 Batch 时从硬盘读取数据，
    从而支持极大的 count 设置而不占用内存。
    """
    def __init__(
        self,
        embeds_c2s: np.ndarray, # 支持 np.memmap
        embeds_s2c: np.ndarray, # 支持 np.memmap
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        count: int = sys.maxsize,
        count2: int = sys.maxsize,
        target_label: int = 0,
        random_state: int = 42
    ):
        # 输入校验
        assert len(embeds_c2s) == len(embeds_s2c) == len(labels), "样本数量不匹配"
        assert batch_size > 0, "batch_size必须大于0"
        
        # 仅仅保存引用（如果是 memmap，这里不占内存）
        self.embeds_c2s = embeds_c2s
        self.embeds_s2c = embeds_s2c
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.count = count
        self.count2 = count2
        self.target_label = target_label
        self.random_state = random_state
        
        # 设置随机种子
        np.random.seed(random_state)
        
        # --- 核心修改 1: 只采样索引，不读取数据 ---
        self.sampled_indices = self._sample_indices()
        
        self.n_samples = len(self.sampled_indices)
        self._iterator_idx = 0
    
    def _sample_indices(self) -> np.ndarray:
        """按规则采样索引（不接触实际特征数据）"""
        # 获取所有标签对应的索引
        all_indices = np.arange(len(self.labels))
        
        # 分离目标标签和其他标签的索引
        target_mask = (self.labels == self.target_label)
        target_indices = all_indices[target_mask]
        
        other_mask = (self.labels != self.target_label)
        other_indices = all_indices[other_mask]
        
        # 1. 采样目标标签索引
        final_target_indices = []
        if len(target_indices) > 0:
            take_target = min(self.count, len(target_indices))
            if take_target > 0:
                # 随机选择索引
                final_target_indices = np.random.choice(
                    target_indices, take_target, replace=False
                )
        
        # 2. 采样其他标签索引
        final_other_indices = []
        if len(other_indices) > 0:
            # 获取其他类别列表
            other_labels = self.labels[other_indices]
            unique_other_labels = np.unique(other_labels)
            
            for label in unique_other_labels:
                # 找到该类别的原始索引
                # 注意：这里需要在全局 all_indices 中找，或者在 other_indices 中找再映射回去
                # 更简单的方法是直接在全局找
                label_global_indices = all_indices[self.labels == label]
                
                if len(label_global_indices) == 0:
                    continue
                
                take = min(self.count2, len(label_global_indices))
                if take > 0:
                    sampled = np.random.choice(
                        label_global_indices, take, replace=False
                    )
                    final_other_indices.extend(sampled)
        
        # 合并所有选中的索引
        if len(final_target_indices) == 0:
            final_indices = np.array(final_other_indices, dtype=int)
        elif len(final_other_indices) == 0:
            final_indices = final_target_indices
        else:
            final_indices = np.concatenate([final_target_indices, final_other_indices])
            
        return final_indices.astype(int)

    def __iter__(self):
        """迭代器初始化"""
        self._iterator_idx = 0
        # 打乱索引（如果需要）
        # 核心修改 2: 只打乱整数索引，速度快且不占内存
        if self.shuffle:
            self.sampled_indices = sk_shuffle(self.sampled_indices, random_state=self.random_state)
        
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取下一个批次"""
        if self._iterator_idx >= self.n_samples:
            raise StopIteration
        
        # 计算当前批次的起止点
        end_idx = self._iterator_idx + self.batch_size
        
        # 处理drop_last逻辑
        if self.drop_last and end_idx > self.n_samples:
            raise StopIteration
        
        # 截取当前批次需要的“全局索引”
        # 注意：这里不要越界，如果不是 drop_last，end_idx 需要截断
        real_end_idx = min(end_idx, self.n_samples)
        batch_indices_ptr = self.sampled_indices[self._iterator_idx : real_end_idx]
        
        self._iterator_idx = end_idx
        
        # --- 核心修改 3: 此时此刻才从 memmap 读取数据 ---
        # Numpy 的 Fancy Indexing 支持 array[list_of_indices]
        # 这一步会触发磁盘 I/O，只读取这一个 batch 的数据到内存
        
        # 必须确保 batch_indices_ptr 是整数数组
        batch_c2s = self.embeds_c2s[batch_indices_ptr]
        batch_s2c = self.embeds_s2c[batch_indices_ptr]
        batch_labels = self.labels[batch_indices_ptr]
        
        # 如果从 memmap 读取，通常已经是 numpy array，不需要 detach
        # 增加鲁棒性检查：如果是 Tensor 则 detach，如果是 numpy 则直接用
        if hasattr(batch_c2s, 'detach'):
            batch_c2s = batch_c2s.detach().cpu().numpy()
        if hasattr(batch_s2c, 'detach'):
            batch_s2c = batch_s2c.detach().cpu().numpy()
        if hasattr(batch_labels, 'detach'):
            batch_labels = batch_labels.detach().cpu().numpy()

        return batch_c2s, batch_s2c, batch_labels

    def __len__(self) -> int:
        """返回批次数量"""
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size


class MyDataLoader:
    def __init__(
        self,
        embeds_c2s: np.ndarray,
        embeds_s2c: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        distances: np.ndarray = None,  # 必须传入
        shuffle: bool = False,
        drop_last: bool = False,
        count: int = sys.maxsize,
        count2: int = sys.maxsize,
        target_label: int = 0,
        random_state: int = 42
    ):
        self.embeds_c2s = embeds_c2s
        self.embeds_s2c = embeds_s2c
        self.labels = labels
        self.distances = distances  # 接收距离
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.count = count
        self.count2 = count2
        self.target_label = target_label
        self.random_state = random_state
        
        random.seed(random_state)
        np.random.seed(random_state)
        
        if self.distances is None:
             print("Warning: MyDataLoader 未接收到 distances, 将无法进行基于距离的优先采样！")

        self.data = self._sample_data()
        self.n_samples = len(self.data)
        self.indices = np.arange(self.n_samples)
        self._iterator_idx = 0

    def _get_priority_indices(self, available_indices, needed_count):
        """
        根据距离优先采样，如果距离未提供则随机采样
        """
        if self.distances is None:
            # 降级为随机采样
            return np.random.choice(available_indices, needed_count, replace=False)
            
        # 1. 获取这些样本对应的距离
        dists = self.distances[available_indices]
        
        # 2. 按距离从小到大排序
        sorted_arg = np.argsort(dists)
        sorted_indices = available_indices[sorted_arg]
        
        # 3. 直接取前 needed_count 个 (即最近的)
        # 如果 available 数量不够，切片会自动处理取全部
        return sorted_indices[:needed_count]

    def _sample_data(self):
        target_indices = np.where(self.labels == self.target_label)[0]
        other_indices = np.where(self.labels != self.target_label)[0]
        other_unique_labels = np.unique(self.labels[other_indices])
        
        sampled_data_list = []
        
        # --- 1. 采样目标标签 ---
        if self.count > 0 and len(target_indices) > 0:
            take_target = min(self.count, len(target_indices))
            sampled_target_idx = self._get_priority_indices(target_indices, take_target)
            
            for i in sampled_target_idx:
                sampled_data_list.append((self.embeds_c2s[i], self.embeds_s2c[i], self.labels[i]))
        
        # --- 2. 采样其他标签 ---
        if self.count2 > 0 and len(other_unique_labels) > 0:
            for label in other_unique_labels:
                label_indices = np.where(self.labels == label)[0]
                if len(label_indices) == 0: continue
                
                take = min(self.count2, len(label_indices))
                sampled_idx = self._get_priority_indices(label_indices, take)
                
                for i in sampled_idx:
                    sampled_data_list.append((self.embeds_c2s[i], self.embeds_s2c[i], self.labels[i]))
        
        return sampled_data_list

    # __iter__, __next__, __len__ 保持不变
    def __iter__(self):
        self._iterator_idx = 0
        if self.shuffle:
            self.indices = sk_shuffle(self.indices, random_state=self.random_state)
        return self

    def __next__(self):
        if self._iterator_idx >= self.n_samples:
            raise StopIteration
        end_idx = self._iterator_idx + self.batch_size
        if self.drop_last and end_idx > self.n_samples:
            raise StopIteration
        batch_indices = self.indices[self._iterator_idx:end_idx]
        self._iterator_idx = end_idx
        
        batch_items = [self.data[i] for i in batch_indices]
        batch_c2s = np.array([item[0] for item in batch_items])
        batch_s2c = np.array([item[1] for item in batch_items])
        batch_labels = np.array([item[2] for item in batch_items])
        return batch_c2s, batch_s2c, batch_labels

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size



# ------------------------------ 示例使用 ------------------------------
if __name__ == "__main__":
    # 1. 生成模拟数据
    n_samples = 1000  # 样本数量
    n_features_c2s = 128  # c2s特征维度
    n_features_s2c = 256  # s2c特征维度
    n_classes = 10  # 类别数
    
    # 随机生成特征向量
    embeds_c2s = np.random.randn(n_samples, n_features_c2s)
    embeds_s2c = np.random.randn(n_samples, n_features_s2c)
    
    # 2. 均衡KMeans聚类
    labels = balanced_kmeans_clustering(embeds_s2c, n_classes=n_classes)
    
    # 打印聚类结果分布
    print("聚类标签分布：")
    for i in range(n_classes):
        print(f"类别 {i}: {np.sum(labels == i)} 个样本")
    
    # 3. 使用自定义DataLoader
    dataloader = MyDataLoader(
        embeds_c2s=embeds_c2s,
        embeds_s2c=embeds_s2c,
        labels=labels,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        count=50,  # 目标标签采样50个
        count2=10,  # 其他类别每个采样10个
        target_label=0  # 目标标签为0
    )
    
    # 迭代数据
    print(f"\nDataLoader批次数量: {len(dataloader)}")
    for batch_idx, (batch_c2s, batch_s2c, batch_labels) in enumerate(dataloader):
        print(f"\n批次 {batch_idx + 1}:")
        print(f"  c2s形状: {batch_c2s.shape}")
        print(f"  s2c形状: {batch_s2c.shape}")
        print(f"  标签形状: {batch_labels.shape}")
        print(f"  标签分布: {np.bincount(batch_labels, minlength=n_classes)}")
        
        # 只打印前2个批次示例
        if batch_idx >= 1:
            break