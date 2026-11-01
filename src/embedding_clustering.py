import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import time

import warnings
warnings.filterwarnings('ignore')

def find_optimal_k_conservative(embeddings, max_k=20, sample_size=10000, tolerance=0.01):
    """
    保守型 K 值搜索策略：宁大勿小。
    
    参数:
        embeddings: 特征向量 [N, Features]
        max_k: 最大搜索范围
        tolerance: 容忍度 (0.10 表示只要分数不低于最高分的 90%，就优先选更大的 K)
        
    返回:
        recommended_k: 推荐的类别数量 (保证 >= 真实类别)
    """
    start_time = time.time()
    # 1. 降采样以加速计算 (若数据量 > 5000)
    if embeddings.shape[0] > sample_size:
        indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
        data_sample = embeddings[indices]
    else:
        data_sample = embeddings

    scores = []
    # 从 3 开始搜，避开 K=2 这种在数据不平衡时容易出现的伪高分
    k_min = 3
    K_range = list(range(k_min, max_k + 1))
    
    print(f"正在搜索最佳 K 值 (策略: 宁大勿小, 容忍度: {tolerance*100}%)...")
    
    for k in K_range:
        # n_init=3 稍微牺牲一点精度换速度，因为我们只看趋势
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3) 
        labels = kmeans.fit_predict(data_sample)
        
        score = silhouette_score(data_sample, labels)
        scores.append(score)
        # 打印进度，方便观察
        # print(f"  K={k}, Score={score:.4f}")

    scores = np.array(scores)
    
    # 2. 找到绝对分数的最高点 (Peak)
    best_idx = np.argmax(scores)
    peak_k = K_range[best_idx]
    max_score = scores[best_idx]
    
    print(f"  -> 数学最高分位于 K={peak_k} (Score={max_score:.4f})")

    # 3. 执行“右侧容忍”策略
    # 我们只关心比 peak_k 更大的 K
    # 只要 score[k] >= max_score * (1 - tolerance)，我们就接受这个更大的 K
    
    candidate_k = peak_k
    threshold_score = max_score * (1.0 - tolerance)
    
    # 向右遍历 (从 peak_k + 1 开始)
    for i in range(best_idx + 1, len(scores)):
        current_k = K_range[i]
        current_score = scores[i]
        
        if current_score >= threshold_score:
            # 如果分数依然很高，我们就更新候选值为这个更大的 K
            candidate_k = current_k
        else:
            # 一旦发现分数断崖式下跌，或者距离 peak_k 太远，就停止
            # 这里限制一下不要偏离太远，比如最多允许偏离 4 个单位，防止飘到 K=20
            if current_k > peak_k + 4:
                break

    # 4. 强制安全兜底 (Safety Bias)
    # 用户要求：绝对杜绝比真实数量小。
    # 如果 peak_k 本身就是低估了 (例如真实10，算出9)，上面的逻辑可能只会停在 9。
    # 所以：如果没有向右移动过 (candidate_k == peak_k)，我们强制 +1
    # if candidate_k == peak_k:
    #     print(f"  -> 警告: 右侧无高分候选，为确保不欠聚类，强制执行 +1 策略。")
    #     candidate_k = peak_k + 1

    # 5. 边界检查
    if candidate_k > max_k:
        candidate_k = max_k

    print(f"最终推荐 K 值: {candidate_k} (原始峰值: {peak_k})")

    end_time = time.time()
    print(f"find_optimal_k_conservative 耗时: {end_time - start_time} 秒")

    return candidate_k


class EmbeddingClustering:
    def __init__(self, max_clusters=20, random_state=42, balance_tolerance=0.1):
        """
        初始化聚类器
        
        参数:
            max_clusters: 最大可能的聚类数量
            random_state: 随机种子，保证结果可复现
            balance_tolerance: 簇数量均衡容忍度（0-1），表示每个簇数量与平均值的最大偏差比例
        """
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.best_k = None
        self.kmeans = None
        self.gmm = None
        self.labels = None
        self.scaler = StandardScaler()
        self.balance_tolerance = balance_tolerance  # 均衡容忍度
    
    def preprocess_embeddings(self, embeddings):
        """预处理嵌入向量：标准化"""
        return self.scaler.fit_transform(embeddings)
    
    def find_optimal_clusters(self, embeddings):
        """
        寻找最优聚类数量
        
        参数:
            embeddings: 输入的嵌入向量，形状为(n_samples, n_features)
            
        返回:
            best_k: 最优的聚类数量
        """
        # 存储不同评估指标的结果
        sse = []          # 误差平方和
        silhouette_scores = []  # 轮廓系数
        calinski_scores = []    # Calinski-Harabasz分数
        davies_scores = []      # Davies-Bouldin分数
        
        # 尝试不同的聚类数量
        for k in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
            labels = kmeans.fit_predict(embeddings)
            sse.append(kmeans.inertia_)
            
            # 计算轮廓系数（计算成本较高）
            silhouette_avg = silhouette_score(embeddings, labels)
            silhouette_scores.append(silhouette_avg)
            
            # 计算Calinski-Harabasz分数
            calinski = calinski_harabasz_score(embeddings, labels)
            calinski_scores.append(calinski)
            
            # 计算Davies-Bouldin分数
            davies = davies_bouldin_score(embeddings, labels)
            davies_scores.append(davies)
        
        # 可视化评估指标（可选）
        # self._plot_metrics(sse, silhouette_scores, calinski_scores, davies_scores)
        
        # 综合多种指标确定最佳聚类数量
        scores = []
        for k_idx in range(len(silhouette_scores)):
            k = k_idx + 2
            # 标准化各项指标
            sil_score = (silhouette_scores[k_idx] - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores) + 1e-8)
            cal_score = (calinski_scores[k_idx] - min(calinski_scores)) / (max(calinski_scores) - min(calinski_scores) + 1e-8)
            dav_score = 1 - (davies_scores[k_idx] - min(davies_scores)) / (max(davies_scores) - min(davies_scores) + 1e-8)
            
            # 加权求和
            weighted_score = 0.4 * sil_score + 0.3 * cal_score + 0.3 * dav_score
            scores.append((k, weighted_score))
        
        # 找到得分最高的聚类数量
        self.best_k = max(scores, key=lambda x: x[1])[0]
        print(f"最佳聚类数量: {self.best_k}")
        
        return self.best_k
    
    def _plot_metrics(self, sse, silhouette_scores, calinski_scores, davies_scores):
        """可视化不同聚类数量下的评估指标"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        k_range = range(2, self.max_clusters + 1)
        
        # 误差平方和 (Elbow Method)
        axes[0, 0].plot(k_range, sse, 'bo-')
        axes[0, 0].set_title('误差平方和 (SSE)')
        axes[0, 0].set_xlabel('聚类数量')
        axes[0, 0].set_ylabel('SSE')
        
        # 轮廓系数
        axes[0, 1].plot(k_range, silhouette_scores, 'ro-')
        axes[0, 1].set_title('轮廓系数 (Silhouette Score)')
        axes[0, 1].set_xlabel('聚类数量')
        axes[0, 1].set_ylabel('分数')
        
        # Calinski-Harabasz分数
        axes[1, 0].plot(k_range, calinski_scores, 'go-')
        axes[1, 0].set_title('Calinski-Harabasz分数')
        axes[1, 0].set_xlabel('聚类数量')
        axes[1, 0].set_ylabel('分数')
        
        # Davies-Bouldin分数
        axes[1, 1].plot(k_range, davies_scores, 'mo-')
        axes[1, 1].set_title('Davies-Bouldin分数')
        axes[1, 1].set_xlabel('聚类数量')
        axes[1, 1].set_ylabel('分数')
        
        plt.tight_layout()
        plt.show()
    
    def _balance_cluster_labels(self, embeddings, initial_labels, n_clusters):
        """
        均衡化聚类标签，确保每个簇的样本数量接近平均值
        
        参数:
            embeddings: 输入嵌入向量 (n_samples, n_features)
            initial_labels: 初始聚类标签
            n_clusters: 聚类数量
            
        返回:
            balanced_labels: 均衡化后的标签
        """
        n_samples = len(embeddings)
        target_samples_per_cluster = n_samples / n_clusters  # 目标平均数量
        max_deviation = target_samples_per_cluster * self.balance_tolerance  # 最大允许偏差
        
        # 计算当前每个簇的样本数量
        cluster_counts = np.bincount(initial_labels, minlength=n_clusters)
        print(f"初始簇数量分布: {cluster_counts}")
        
        # 计算每个簇的偏差
        cluster_deviations = cluster_counts - target_samples_per_cluster
        
        # 复制初始标签用于修改
        balanced_labels = initial_labels.copy()
        
        # 迭代调整：将超量簇的样本分配到不足的簇
        max_iter = 10  # 最大迭代次数
        for _ in range(max_iter):
            # 找到超量的簇（正偏差）和不足的簇（负偏差）
            over_clusters = [i for i in range(n_clusters) if cluster_deviations[i] > max_deviation]
            under_clusters = [i for i in range(n_clusters) if cluster_deviations[i] < -max_deviation]
            
            if not over_clusters or not under_clusters:
                break  # 已达到均衡要求
            
            # 处理每个超量簇
            for over_cluster in over_clusters:
                # 找到该簇的所有样本索引
                over_sample_indices = np.where(balanced_labels == over_cluster)[0]
                np.random.shuffle(over_sample_indices)  # 随机打乱
                
                # 需要移出的样本数量
                need_to_move = int(cluster_deviations[over_cluster] - max_deviation)
                if need_to_move <= 0:
                    continue
                
                # 为每个不足的簇分配样本
                for under_cluster in under_clusters:
                    if need_to_move <= 0:
                        break
                    
                    # 计算该不足簇需要补充的样本数量
                    need_to_add = int(-cluster_deviations[under_cluster] - max_deviation)
                    if need_to_add <= 0:
                        continue
                    
                    # 实际要移动的样本数量
                    move_count = min(need_to_move, need_to_add)
                    
                    # 选择要移动的样本（优先选择与当前簇中心距离最远的样本）
                    # 计算样本到超量簇中心的距离
                    over_center = embeddings[balanced_labels == over_cluster].mean(axis=0)
                    sample_distances = np.linalg.norm(embeddings[over_sample_indices] - over_center, axis=1)
                    # 选择距离最远的样本（这些样本对当前簇的归属度最低）
                    farthest_indices = np.argsort(sample_distances)[-move_count:]
                    samples_to_move = over_sample_indices[farthest_indices]
                    
                    # 更新标签
                    balanced_labels[samples_to_move] = under_cluster
                    
                    # 更新计数和偏差
                    cluster_counts[over_cluster] -= move_count
                    cluster_counts[under_cluster] += move_count
                    cluster_deviations[over_cluster] = cluster_counts[over_cluster] - target_samples_per_cluster
                    cluster_deviations[under_cluster] = cluster_counts[under_cluster] - target_samples_per_cluster
                    
                    need_to_move -= move_count
            
        print(f"均衡后簇数量分布: {cluster_counts}")
        return balanced_labels
    
    def cluster(self, embeddings, use_gmm=True):
        """
        对嵌入向量进行聚类（带数量均衡约束）
        
        参数:
            embeddings: 输入的嵌入向量，形状为(n_samples, n_features)
            use_gmm: 是否使用GMM聚类（False则使用KMeans）
            
        返回:
            labels: 均衡化后的聚类标签
        """
        # 预处理（可选，注释掉则使用原始embeddings）
        # processed_embeddings = self.preprocess_embeddings(embeddings)
        processed_embeddings = embeddings
        
        if self.best_k is None:
            self.find_optimal_clusters(processed_embeddings)
        
        # 第一步：执行初始聚类
        if use_gmm:
            # GMM聚类
            self.gmm = GaussianMixture(
                n_components=self.best_k, 
                covariance_type='full', 
                random_state=self.random_state, 
                reg_covar=1e-6
            )
            processed_embeddings = processed_embeddings.astype(np.float64)
            self.gmm.fit(processed_embeddings)
            initial_labels = self.gmm.predict(processed_embeddings)
        else:
            # KMeans聚类
            self.kmeans = KMeans(
                n_clusters=self.best_k, 
                random_state=self.random_state, 
                n_init='auto'
            )
            self.kmeans.fit(processed_embeddings)
            initial_labels = self.kmeans.predict(processed_embeddings)
        
        # 第二步：均衡化聚类标签
        self.labels = self._balance_cluster_labels(
            processed_embeddings, 
            initial_labels, 
            self.best_k
        )
        
        # 验证最终分布
        final_counts = np.bincount(self.labels, minlength=self.best_k)
        print(f"最终簇数量分布: {final_counts}")
        print(f"簇数量标准差: {np.std(final_counts):.2f} (越小越均衡)")
        
        return self.labels
    
    def visualize_clusters(self, embeddings, labels=None, save_path=None):
        """
        可视化聚类结果（使用PCA降维到2D）
        
        参数:
            embeddings: 输入的嵌入向量
            labels: 聚类标签，如果为None则使用已聚类的标签
        """
        # 预处理
        processed_embeddings = self.preprocess_embeddings(embeddings)
        
        # 如果没有标签，则使用已聚类的标签
        if labels is None:
            labels = self.labels
        
        # 使用PCA降维到2D以便可视化
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(processed_embeddings)
        
        # 绘制聚类结果
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, 
                             cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(scatter, label='聚类标签')
        plt.title(f'图像Embeddings聚类可视化 (聚类数量: {self.best_k})')
        plt.xlabel('PCA维度 1')
        plt.ylabel('PCA维度 2')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 如果指定了保存路径，则保存图片
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"聚类可视化图片已保存到: {save_path}")
        plt.show()

    def find_nearest_cluster_label(self, target_label):
        """找到距离目标标签对应的中心最近的簇中心"""
        if self.kmeans is None:
            raise ValueError("请先执行KMeans聚类（use_gmm=False）")
            
        target_center = self.kmeans.cluster_centers_[target_label]
        all_centers = self.kmeans.cluster_centers_
        
        # 计算目标中心与所有其他中心的欧氏距离
        ret = None
        distances = []
        for i, center in enumerate(all_centers):
            if i != target_label:  # 排除目标中心本身
                distance = np.linalg.norm(target_center - center)
                distances.append((i, distance))
        
        # 找到距离最近的簇中心
        if distances:
            nearest_cluster_idx, nearest_distance = min(distances, key=lambda x: x[1])
            ret = nearest_cluster_idx
            print(f"距离标签 {target_label} 最近的簇中心是 {nearest_cluster_idx}，距离为 {nearest_distance:.4f}")
        else:
            print("只有一个簇中心，无法找到最近的簇中心")
        
        return ret

    def find_farthest_cluster_label(self, target_label):
        """找到距离目标标签对应的中心最远的簇中心"""
        if self.kmeans is None:
            raise ValueError("请先执行KMeans聚类（use_gmm=False）")
            
        target_center = self.kmeans.cluster_centers_[target_label]
        all_centers = self.kmeans.cluster_centers_
        
        # 计算目标中心与所有其他中心的欧氏距离
        ret = None
        distances = []
        for i, center in enumerate(all_centers):
            if i != target_label:  # 排除目标中心本身
                distance = np.linalg.norm(target_center - center)
                distances.append((i, distance))
        
        # 找到距离最远的簇中心
        if distances:
            farthest_cluster_idx, farthest_distance = max(distances, key=lambda x: x[1])
            ret = farthest_cluster_idx
            print(f"距离标签 {target_label} 最远的簇中心是 {farthest_cluster_idx}，距离为 {farthest_distance:.4f}")
        else:
            print("只有一个簇中心，无法找到最远的簇中心")
        
        return ret

# 示例用法
if __name__ == "__main__":
    # 生成模拟的ResNet-50倒数第二层输出的Embeddings
    np.random.seed(42)
    num_samples = 1500
    embedding_dim = 2048
    num_true_classes = 20
    samples_per_class = num_samples // num_true_classes
    
    # 生成模拟数据
    embeddings = []
    for i in range(num_true_classes):
        # 为每个类别生成有细微差别的中心
        center = np.random.randn(embedding_dim) * 2
        # 围绕中心生成样本
        class_embeddings = np.random.randn(samples_per_class, embedding_dim) + center
        embeddings.append(class_embeddings)
    
    # 合并所有样本
    embeddings = np.vstack(embeddings)
    print(f"生成模拟Embeddings: 形状 {embeddings.shape}")
    
    # 初始化聚类器，设置均衡容忍度为10%
    clustering = EmbeddingClustering(
        max_clusters=30, 
        random_state=42,
        balance_tolerance=0.1  # 每个簇数量与平均值的偏差不超过10%
    )
    
    # 执行聚类（使用GMM+均衡化）
    cluster_labels = clustering.cluster(embeddings, use_gmm=True)
    
    # 可视化聚类结果
    clustering.visualize_clusters(embeddings, save_path="balanced_clusters.png")
    
    print(f"\n聚类完成，得到 {clustering.best_k} 个聚类")
    print(f"实际类别数量为 {num_true_classes}")
    print(f"最终簇数量分布: {np.bincount(cluster_labels)}")
    print(f"簇数量变异系数: {np.std(np.bincount(cluster_labels))/np.mean(np.bincount(cluster_labels)):.2%}")