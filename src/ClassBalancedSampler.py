import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets, transforms
import os


# 作用：按指定数量加载各类别样本
# target_label：指定要单独控制样本数的类别（非负数有效）
# count：target_label类别的样本数
# count2：其他类别的样本数
class ClassBalancedSampler(Sampler):
    def __init__(self, dataset, count, count2 ,target_label=-1, shuffle=True):
        self.count = count  # 仅对target_label类别生效：指定该类别要选取的样本数
        self.count2 = count2        # 其他类别的样本数
        self.shuffle = shuffle  # 是否打乱最终的索引顺序
        self.target_label = target_label  # 需要限制样本数的类别

        # 1. 动态获取数据集的类别总数
        if hasattr(dataset, 'class_to_idx'):
            self.num_classes = len(dataset.class_to_idx)
        elif hasattr(dataset, 'num_classes'):
            self.num_classes = dataset.num_classes
        elif hasattr(dataset, 'classes'):
            self.num_classes = len(dataset.classes)
        elif hasattr(dataset, 'targets'):
            self.num_classes = len(set(dataset.targets))
        else:
            all_targets = [target for _, target in dataset]
            self.num_classes = len(set(all_targets))

        # 2. 统计每个类别的样本索引
        class_indices = {cls: [] for cls in range(self.num_classes)}
        for idx, (_, target) in enumerate(dataset):
            target = int(target)
            class_indices[target].append(idx)

        # 3. 筛选样本索引：target_label类取count个，其他类取全部
        self.indices = []
        
        for cls in range(self.num_classes):
            if cls == self.target_label:
                # 仅对target_label类别限制样本数（不足count则取全部）
                selected = class_indices[cls][:count]
                print(f"类别 {cls}（限制样本数）：共{len(class_indices[cls])}个，选取{len(selected)}个")
            else:
                # 其他类别取全部样本
                selected = class_indices[cls][:count2] 
                print(f"类别 {cls}（全部样本）：共{len(class_indices[cls])}个，选取{len(selected)}个")               
            self.indices.extend(selected)

        # 4. 打乱索引顺序
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


# 作用：收集所有类别的样本，target_label指定的类别仅取count个，其他类别取全部
# target_label参数：非负数表示指定要限制样本数的类别；
class ClassBalancedSampler2(Sampler):
    def __init__(self, dataset, count, target_label=-1, shuffle=True):
        self.count = count  # 仅对target_label类别生效：指定该类别要选取的样本数
        self.shuffle = shuffle  # 是否打乱最终的索引顺序
        self.target_label = target_label  # 需要限制样本数的类别

        # 1. 动态获取数据集的类别总数
        if hasattr(dataset, 'class_to_idx'):
            self.num_classes = len(dataset.class_to_idx)
        elif hasattr(dataset, 'num_classes'):
            self.num_classes = dataset.num_classes
        elif hasattr(dataset, 'classes'):
            self.num_classes = len(dataset.classes)
        elif hasattr(dataset, 'targets'):
            self.num_classes = len(set(dataset.targets))
        else:
            all_targets = [target for _, target in dataset]
            self.num_classes = len(set(all_targets))

        # 2. 统计每个类别的样本索引
        class_indices = {cls: [] for cls in range(self.num_classes)}
        for idx, (_, target) in enumerate(dataset):
            target = int(target)
            class_indices[target].append(idx)

        # 3. 筛选样本索引：target_label类取count个，其他类取全部
        self.indices = []
        
        for cls in range(self.num_classes):
            if cls == self.target_label:
                # 仅对target_label类别限制样本数（不足count则取全部）
                selected = class_indices[cls][:count]
                print(f"类别 {cls}（限制样本数）：共{len(class_indices[cls])}个，选取{len(selected)}个")
            else:
                # 其他类别取全部样本
                selected = class_indices[cls] 
                print(f"类别 {cls}（全部样本）：共{len(selected)}个")               
            self.indices.extend(selected)

        # 4. 打乱索引顺序
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


#作用：每个类别要选取count个样本，       
# target_label参数：如果该参数为非负数  ，在收集数据集各个类别的样本的时候，不收集标签为target_label的样本，  如果该参数为负数则还是收集所有类别的样本
class ClassBalancedSampler3(Sampler):
    # 新增参数 target_label，默认为 -1，表示不排除任何类别
    def __init__(self, dataset, count, target_label=-1, shuffle=True):
        self.count = count  # 每个类别要选取的样本数
        self.shuffle = shuffle  # 是否打乱最终的索引顺序
        self.target_label = target_label  # 需要排除的类别标签

        # 1. 动态获取数据集的类别总数
        if hasattr(dataset, 'class_to_idx'):
            self.num_classes = len(dataset.class_to_idx)
        elif hasattr(dataset, 'num_classes'):
            self.num_classes = dataset.num_classes
        elif hasattr(dataset, 'classes'):
            self.num_classes = len(dataset.classes)
        elif hasattr(dataset, 'targets'):
            self.num_classes = len(set(dataset.targets))
        else:
            all_targets = [target for _, target in dataset]
            self.num_classes = len(set(all_targets))

        # 2. 统计每个类别的样本索引
        class_indices = {cls: [] for cls in range(self.num_classes)}
        for idx, (_, target) in enumerate(dataset):
            target = int(target)
            # 核心改动：如果当前类别是要排除的类别，则跳过
            if target == self.target_label:
                continue
            class_indices[target].append(idx)

        # 3. 筛选每个类别前count个索引，合并成总索引列表
        self.indices = []
        # 遍历所有类别，但跳过要排除的类别
        for cls in range(self.num_classes):
            if cls == self.target_label:
                continue
            selected = class_indices[cls][:count]
            self.indices.extend(selected)

        # 4. 打乱索引顺序
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


# 测试代码（验证功能正确性）
if __name__ == "__main__":
    # 数据预处理（CIFAR-10标准预处理）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 数据集路径（自动适配当前文件上级目录的data文件夹）
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    root_dir = os.path.join(base_dir, "data")

    # 加载CIFAR-10训练集（共10类，每类5000个样本）
    full_dataset = datasets.CIFAR10(
        root=root_dir,
        train=True,
        transform=transform,
        download=False  # 已下载过数据集则设为False，首次使用设为True
    )

    # ---------------------- 测试场景：单独控制类别5（狗）的样本数 ----------------------
    print("=" * 60)
    print("测试场景：target_label=5（狗），count=200（狗类取200个），count2=100（其他类各取100个）")
    print("=" * 60)
    
    # 创建采样器
    sampler = ClassBalancedSampler(
        dataset=full_dataset,
        count=200,       # 狗类（类别5）取200个
        count2=100,      # 其他9类各取100个
        target_label=5,  # 单独控制的类别：狗（CIFAR-10中类别5对应狗）
        shuffle=True     # 打乱顺序
    )

    # 创建DataLoader
    train_loader = DataLoader(
        dataset=full_dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=2  # 根据CPU核心数调整，Windows系统可设为0
    )

    # 验证加载结果（统计各类别实际加载数量）
    print(f"\n总加载样本数：{len(sampler)}")  # 预期：200 + 9*100 = 1100
    print("\n各类别实际加载数量明细：")
    
    class_counts = np.zeros(10, dtype=int)
    for _, targets in train_loader:
        for target in targets:
            class_counts[target.item()] += 1

    for cls_idx, cnt in enumerate(class_counts):
        print(f"类别 {cls_idx} ({full_dataset.classes[cls_idx]}): {cnt} 个样本")