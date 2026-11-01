import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import os

# 设置绘图风格，使其符合学术论文规范
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

model_name = 'vgg16'
dataset = 'cifar10'
sb = 10

def calculate_silhouette_curve(data, k_min=2, k_max=15, sample_size=10000):
    """
    计算不同 K 值下的轮廓系数
    对应论文公式: S_k = (b(i) - a(i)) / max(a(i), b(i)) [cite: 68]
    """
    scores = []
    k_range = list(range(k_min, k_max + 1))
    
    # 如果数据量太大，进行降采样以加速计算 (参考你的 find_optimal_k_conservative 函数)
    if data.shape[0] > sample_size:
        print(f"数据量较大 ({data.shape[0]})，进行降采样至 {sample_size} 用于轮廓系数计算...")
        indices = np.random.choice(data.shape[0], sample_size, replace=False)
        data_sample = data[indices]
    else:
        data_sample = data

    print(f"正在计算 K={k_min} 到 K={k_max} 的轮廓系数...")
    
    for k in k_range:
        # 使用 KMeans 进行聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_sample)
        
        # 计算轮廓系数
        score = silhouette_score(data_sample, labels)
        scores.append(score)
        print(f"  K={k}: Silhouette Score={score:.4f}")
        
    return k_range, scores

def plot_k_estimation(k_range, scores, true_k=None, save_path=None):
    """
    绘制 K 值估计对比图
    """
    k_range = np.array(k_range)
    scores = np.array(scores)
    
    # 找到峰值对应的 K (即 Estimated K*)
    best_idx = np.argmax(scores)
    estimated_k = k_range[best_idx]
    max_score = scores[best_idx]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(k_range, scores, marker='o', color='royalblue', linewidth=2, label='Silhouette Score')
    
    # 标记 Estimated K* (峰值)
    plt.axvline(x=estimated_k, color='red', linestyle='--', alpha=0.8, 
                label=f'Estimated $K^*={estimated_k}$')
    plt.scatter([estimated_k], [max_score], color='red', zorder=5, s=100)
    
    # 标记 Ground Truth K (如果已知)
    if true_k is not None:
        plt.axvline(x=true_k, color='green', linestyle=':', linewidth=2, alpha=0.8, 
                    label=f'Ground Truth $K_{{gt}}={true_k}$')
    
    # 图表装饰
    #plt.title('Optimal $K$ Estimation via Silhouette Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Clusters ($k$)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.xticks(k_range)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图表已保存至: {save_path}")
    
    plt.show()

# ==========================================
# 场景 1: 使用模拟数据运行 (Demo)
# 对应论文实验部分: 验证算法在理想情况下的有效性
# ==========================================
def run_demo():
    print("--- 运行模拟数据实验 ---")
    # 模拟 CIFAR-10 特征空间: 10个类别，特征维度64，样本数2000
    K_gt = 10
    X, y = make_blobs(n_samples=2000, n_features=64, centers=K_gt, 
                      cluster_std=2.0, random_state=9202)
    
    # 计算曲线
    k_range, scores = calculate_silhouette_curve(X, k_min=3, k_max=15)
    
    # 绘图
    plot_k_estimation(k_range, scores, true_k=K_gt, save_path='k_estimation_demo.png')

# ==========================================
# 场景 2: 使用你的真实训练数据 (.npy)
# 对应代码中的: embeds_s2c_cifar10_sb1.npy
# ==========================================
def run_with_real_data():
    print("\n--- 运行真实数据实验 ---")
    
    # 1. 路径设置 (请根据你的实际路径修改)

    # 假设文件在你代码中定义的 save_path930/embeddings 下
    file_path = f"./saved_model930/embeddings/embeds_s2c_{dataset}_sb{sb}.npy"
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}，请先运行你的训练代码生成 embedding。")
        return

    # 2. 加载数据 (使用 mmap_mode 防止内存溢出，参考你的 load_embeddings_and_labels)
    embeds_s2c = np.load(file_path, mmap_mode='r')
    print(f"加载数据形状: {embeds_s2c.shape}")
    
    # 3. 展平数据 (N, C, H, W) -> (N, Feature_Dim)
    n_samples = embeds_s2c.shape[0]
    data_flat = embeds_s2c.reshape(n_samples, -1)
    
    # 4. 计算与绘图
    # CIFAR-10 的真实类别是 10
    k_range, scores = calculate_silhouette_curve(data_flat, k_min=2, k_max=20)
    plot_k_estimation(k_range, scores, true_k=10, save_path=f'./exp/k_estimation_{model_name}_sb{sb}_{dataset}.png')

if __name__ == "__main__":
    # 默认运行模拟演示，你可以取消注释下一行来运行真实数据
   # run_demo()
    run_with_real_data()