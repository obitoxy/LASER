import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
import matplotlib.patches as mpatches

# ================= 配置区域 =================
# 如果没有真实数据，设为 True 可以先看效果
DEMO_MODE = True 

# 真实数据路径 (当 DEMO_MODE = False 时生效)
# 请替换为你实际保存的 .npy 文件路径
PATH_CLEAN_EMBEDS = "./saved_model930/embeddings/embeds_s2c_cifar10_sb10_clean.npy"
PATH_NAIVE_EMBEDS = "./saved_model930/embeddings/embeds_s2c_cifar10_sb10_naive.npy" # alpha=0
PATH_LASER_EMBEDS = "./saved_model930/embeddings/embeds_s2c_cifar10_sb10_laser.npy" # alpha=70
PATH_LABELS = "./saved_model930/embeddings/labels_true_cifar10_sb10.npy"

# 实验设置
TARGET_CLASS = 4  # 受害类别 (Victim)
PROXY_CLASS = 5   # 代理类别 (Proxy / Destination)
TOTAL_CLASSES = 10
SAMPLE_SIZE = 2000 # t-SNE 采样点数，避免太慢
RANDOM_SEED = 9202

# ================= 绘图工具函数 =================

def plot_tsne(embeds, labels, title, ax, target_cls, proxy_cls):
    """
    绘制单张 t-SNE 图，突出显示 Target 和 Proxy
    """
    print(f"正在计算 t-SNE: {title} ...")
    
    # 1. 降维
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=RANDOM_SEED)
    embeds_2d = tsne.fit_transform(embeds)
    
    # 2. 颜色定义
    #以此区分：受害类(红)，代理类(蓝)，其他类(灰)
    colors = []
    alphas = []
    sizes = []
    zorders = []
    
    for label in labels:
        if label == target_cls:
            colors.append('#D62728') # Red for Victim
            alphas.append(0.9)
            sizes.append(25)
            zorders.append(10) # Top layer
        elif label == proxy_cls:
            colors.append('#1F77B4') # Blue for Proxy
            alphas.append(0.6)
            sizes.append(25)
            zorders.append(5)
        else:
            colors.append('#C7C7C7') # Grey for others
            alphas.append(0.2)
            sizes.append(10)
            zorders.append(1) # Bottom layer

    # 3. 散点图
    ax.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=colors, s=sizes, alpha=alphas, zorder=1)
    
    # 4. 装饰
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks([]) # 移除坐标轴刻度，t-SNE 坐标值无物理意义
    ax.set_yticks([])
    
    # 5. 自定义图例
    legend_patches = [
        mpatches.Patch(color='#D62728', label=f'Victim (Class {target_cls})'),
        mpatches.Patch(color='#1F77B4', label=f'Proxy (Class {proxy_cls})'),
        mpatches.Patch(color='#C7C7C7', label='Non-victim Classes')
    ]
    ax.legend(handles=legend_patches, loc='lower right', frameon=True)

def plot_confusion_matrix_custom(y_true, y_pred, title, ax, target_cls):
    """
    绘制混淆矩阵热力图
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(TOTAL_CLASSES)))
    # 归一化 (按行)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=False, fmt=".2f", cmap="Blues", cbar=True, ax=ax, vmin=0, vmax=1)
    
    # 高亮受害类别的行
    # 在 Target Class 对应的行画一个红框
    ax.add_patch(plt.Rectangle((0, target_cls), TOTAL_CLASSES, 1, fill=False, edgecolor='red', lw=2, clip_on=False))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

# ================= 模拟数据生成器 (Demo Only) =================
def generate_demo_data(n_samples=2000, n_features=64, n_classes=10):
    # 1. 基础数据 (Separated)
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, cluster_std=2.0, random_state=RANDOM_SEED)
    
    # 模拟 Clean
    X_clean = X.copy()
    
    # 模拟 Naive Attack (No Anchor -> Catastrophic Forgetting)
    # 所有类别的方差变大，且稍微发生位移，模拟特征空间坍塌
    X_naive = X.copy()
    noise = np.random.normal(0, 8.0, X_naive.shape) # 巨大的噪声
    X_naive += noise 
    
    # 模拟 LASER Attack (Anchored -> Targeted Pulling)
    # 非受害类保持不变，受害类(Target)移动到代理类(Proxy)附近
    X_laser = X.copy()
    target_indices = np.where(y == TARGET_CLASS)[0]
    proxy_indices = np.where(y == PROXY_CLASS)[0]
    
    # 计算 Proxy 的中心
    proxy_center = np.mean(X[proxy_indices], axis=0)
    
    # 将 Target 拉向 Proxy (保留一点点自身的方差，模拟拉长效果)
    # 模拟 "Trailing" 效果：一部分完全重叠，一部分还在路上
    for idx in target_indices:
        # 插值移动：90% 靠近 Proxy
        X_laser[idx] = X_laser[idx] * 0.1 + proxy_center * 0.9 + np.random.normal(0, 1.0, n_features)

    return X_clean, X_naive, X_laser, y

# ================= 主程序 =================

def main():
    # 1. 数据加载
    if DEMO_MODE:
        print("--- 正在运行演示模式 (生成模拟数据) ---")
        embeds_clean, embeds_naive, embeds_laser, y_true = generate_demo_data()
        # 模拟预测结果用于混淆矩阵
        # Clean: 准确率很高
        y_pred_clean = y_true.copy()
        # LASER: Target 变成 Proxy，其他不变
        y_pred_laser = y_true.copy()
        y_pred_laser[y_true == TARGET_CLASS] = PROXY_CLASS
    else:
        print("--- 正在加载真实实验数据 ---")
        # 加载并降维/采样 (如果数据量大)
        def load_and_sample(path, labels_path):
            data = np.load(path, mmap_mode='r')
            lbls = np.load(labels_path, mmap_mode='r')
            if data.shape[0] > SAMPLE_SIZE:
                indices = np.random.choice(data.shape[0], SAMPLE_SIZE, replace=False)
                return data[indices], lbls[indices]
            return data, lbls

        embeds_clean, y_true_sample = load_and_sample(PATH_CLEAN_EMBEDS, PATH_LABELS)
        embeds_naive, _ = load_and_sample(PATH_NAIVE_EMBEDS, PATH_LABELS)
        embeds_laser, _ = load_and_sample(PATH_LASER_EMBEDS, PATH_LABELS)
        
        # 注意：真实混淆矩阵需要模型的预测输出 (logits.argmax)，
        # 这里假设你已经有了预测标签，如果没有，可以使用 KMeans 对 embeds 进行聚类来近似模拟预测
        # 或者你需要保存 model 的预测结果。这里暂用真实标签模拟 Clean，用 embeds 距离模拟 Attack。
        y_pred_clean = y_true_sample # 简化
        y_pred_laser = y_true_sample.copy() 
        # 这里只是占位，真实实验请加载真实的 predictions
        
    # ================= 任务 1: t-SNE 可视化 (1行3列) =================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图 A: Attack Before (Clean)
    plot_tsne(embeds_clean, y_true, "Fig A: Attack Before (Clean)", axes[0], TARGET_CLASS, PROXY_CLASS)
    
    # 图 B: Naive Attack (No Anchors)
    plot_tsne(embeds_naive, y_true, "Fig B: Naive Attack (No Anchors)", axes[1], TARGET_CLASS, PROXY_CLASS)
    
    # 图 C: LASER Attack (Ours)
    plot_tsne(embeds_laser, y_true, "Fig C: LASER Attack (Ours)", axes[2], TARGET_CLASS, PROXY_CLASS)
    
    plt.tight_layout()
    plt.savefig("./exp/experiment_tsne_comparison.png", dpi=300)
    print("t-SNE 对比图已保存: experiment_tsne_comparison.png")
    plt.show()

    # ================= 任务 2: 混淆矩阵 (1行2列) =================
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    plot_confusion_matrix_custom(y_true, y_pred_clean, "Confusion Matrix (Before Attack)", axes2[0], TARGET_CLASS)
    plot_confusion_matrix_custom(y_true, y_pred_laser, "Confusion Matrix (After LASER)", axes2[1], TARGET_CLASS)
    
    plt.tight_layout()
    plt.savefig("./exp/experiment_confusion_matrix.png", dpi=300)
    print("混淆矩阵图已保存: experiment_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    main()