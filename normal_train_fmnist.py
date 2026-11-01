import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入你的模型定义 (请确保文件路径正确)
from src.models.vgg16 import NetClient1, NetServer, NetClient2

# ==========================================
# 1. 极速环境配置 (4090D 专属)
# ==========================================
# 优化显存分配，防止碎片化 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 开启 TF32 矩阵加速 (Ampere/Ada 架构神器)
torch.set_float32_matmul_precision('high')
# 开启 CuDNN Benchmark (输入尺寸固定时加速明显)
torch.backends.cudnn.benchmark = True

# ==========================================
# 2. 辅助函数：初始化与 Mixup
# ==========================================

def weights_init(m):
    '''VGG 必须进行 Kaiming 初始化，否则很难收敛'''
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Mixup 数据增强'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 3. 训练主程序
# ==========================================

def train():
    # --- 强力配置 ---
    BATCH_SIZE = 512       # 4090D 显存充足，512 是兼顾速度和收敛的甜点
    LR = 0.05              # SGD 初始学习率 (配合 Warmup)
    EPOCHS = 300           # VGG 收敛较慢，FashionMNIST 150轮足够
    NUM_WORKERS = 15       # 利用你的 15 vCPU
    SPLIT_BLOCK = 1        # VGG 切割点 (通常选中间层)
    NUM_CLASSES = 10
    DEVICE = torch.device("cuda")
    
    # 定义保存路径
    save_path930 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model930")
    save_dir = os.path.join(save_path930, f"normal_model/fmnist/sb{SPLIT_BLOCK}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 Training VGG16 on FashionMNIST | Device: {DEVICE}")
    print(f"📦 Batch: {BATCH_SIZE} | ⚡ AMP: bfloat16 | 🛠️ Mixup: Enabled")

    # --- 数据准备 (SOTA 级增强) ---
    # FashionMNIST 是 28x28 单通道。
    # 为了防止 VGG 过拟合，我们使用比较激进的增强。
    transform_train = transforms.Compose([
        transforms.Resize(32), # 28 -> 32，保留更多特征
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)), # FashionMNIST 均值方差
        transforms.RandomErasing(p=0.3), # 随机擦除，防止过拟合
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    # persistent_workers=True 避免每个 epoch 重建进程
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    
    testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=1024, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    # --- 模型初始化 ---
    # 注意：FashionMNIST 是单通道，in_=1
    net_client1 = NetClient1(in_=1, num_class=NUM_CLASSES, split_block=SPLIT_BLOCK).to(DEVICE)
    net_server = NetServer(num_class=NUM_CLASSES, split_block=SPLIT_BLOCK).to(DEVICE)
    net_client2 = NetClient2(num_class=NUM_CLASSES).to(DEVICE)

    # --- 权重初始化 ---
    print("🎨 Applying Kaiming Initialization...")
    net_client1.apply(weights_init)
    net_server.apply(weights_init)
    # net_client2 内部是预训练的 VGG classifier，最好也重置以匹配从零训练
    net_client2.apply(weights_init)

    # --- 编译加速 ---
    # VGG 是标准线性结构，torch.compile 效果极好且不容易报错
    print("🔥 Compiling models...")
    try:
        net_client1 = torch.compile(net_client1, mode='default')
        net_server = torch.compile(net_server, mode='default')
        net_client2 = torch.compile(net_client2, mode='default')
    except Exception as e:
        print(f"Compile Warning: {e}. Proceeding without compile.")

    # --- 优化器与调度器 ---
    # VGG 这种深层无残差网络，SGD + Momentum 是最佳选择
    params = list(net_client1.parameters()) + list(net_server.parameters()) + list(net_client2.parameters())
    optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # 余弦退火调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        # Warmup: 前5轮线性预热，防止 VGG 梯度爆炸
        if epoch < 5:
            warmup_lr = LR * (epoch + 1) / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        net_client1.train()
        net_server.train()
        net_client2.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", dynamic_ncols=True)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # --- 1. Mixup ---
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, device=DEVICE)

            # --- 2. Forward (Split Learning + AMP) ---
            # 使用 bfloat16
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Client 1
                client1_out = net_client1(inputs)
                # Server
                server_out = net_server(client1_out)
                # Client 2
                outputs = net_client2(server_out)
                
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # --- 3. Backward (Split Learning 手动传递) ---
            optimizer.zero_grad()
            loss.backward()

            # 梯度回传
            grad_to_server = net_client2.client_backward()
            grad_to_client1 = net_server.server_backward(grad_to_server)
            net_client1.client_backward(grad_to_client1)

            # --- 梯度裁剪 (VGG 必备) ---
            # 防止反向传播初期梯度爆炸
            nn.utils.clip_grad_norm_(params, max_norm=5.0)

            optimizer.step()

            # --- 统计 ---
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # Mixup 准确率估算
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            
            pbar.set_postfix({
                "Loss": f"{running_loss/(total/BATCH_SIZE):.3f}", 
                "Acc": f"{100.*correct/total:.2f}%", 
                "LR": f"{optimizer.param_groups[0]['lr']:.4f}"
            })

        # 调度器更新
        if epoch >= 5:
            scheduler.step()

        # --- 验证循环 ---
        net_client1.eval()
        net_server.eval()
        net_client2.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out1 = net_client1(inputs)
                    out2 = net_server(out1)
                    outputs = net_client2(out2)
                
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        acc = 100. * test_correct / test_total
        tqdm.write(f"Epoch {epoch+1} Test Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

        # --- 保存模型 ---
        if acc > best_acc:
            best_acc = acc
            tqdm.write(f"🏆 New Record! Acc: {best_acc:.2f}%")
            
            # 处理 torch.compile 前缀
            c1_state = net_client1._orig_mod.state_dict() if hasattr(net_client1, '_orig_mod') else net_client1.state_dict()
            s_state = net_server._orig_mod.state_dict() if hasattr(net_server, '_orig_mod') else net_server.state_dict()
            c2_state = net_client2._orig_mod.state_dict() if hasattr(net_client2, '_orig_mod') else net_client2.state_dict()
            
            model_path = os.path.join(save_dir, f"net1_{epoch+1}.pt")
            torch.save(c1_state, model_path)
            
            model_path = os.path.join(save_dir, f"server_{epoch+1}.pt")
            torch.save(s_state, model_path)
            
            model_path = os.path.join(save_dir, f"net3_{epoch+1}.pt")
            torch.save(c2_state, model_path)
            
            tqdm.write("Models saved successfully!")

if __name__ == '__main__':
    train()