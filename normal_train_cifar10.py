import os
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import cast
from src.models.vgg16 import NetClient1,NetServer,NetClient2

# ==========================================
# 1. 硬件与环境极速配置
# ==========================================
# 开启 4090D 的 TF32 矩阵加速
torch.set_float32_matmul_precision('high')
# 开启 CuDNN 自动寻找最快算法
torch.backends.cudnn.benchmark = True
# 优化显存分配策略，防止碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
save_path930 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model930")

# ==========================================
# 2. VGG16 模型定义 (保持你的原始定义)
# ==========================================

# ==========================================
# 3. 辅助函数：初始化与Mixup
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
# 4. 主训练循环
# ==========================================

def train():
    # --- 强力参数配置 ---
    # VGG 全连接层参数量巨大 (100MB+), Batch Size 不能像 ResNeXt 开到 1024
    # 512 是 24GB 显存下的安全且高效的 sweet spot
    BATCH_SIZE = 512
    EPOCHS = 400        # VGG 需要较长时间训练
    LR = 0.05           # SGD 初始学习率 (配合 Warmup)
    WEIGHT_DECAY = 5e-4 # 正则化
    SPLIT_BLOCK = 1     # 在第1个卷积块处切割 (VGG共13个卷积块，6大约在中间)
    NUM_CLASSES = 10
    DEVICE = torch.device("cuda")
    save_dir = os.path.join(save_path930, f"normal_model/cifar10/sb{SPLIT_BLOCK}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"🚀 Training VGG16 on {DEVICE} | Batch: {BATCH_SIZE} | AMP: bfloat16")

    # --- 数据增强 (SOTA 标准) ---
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), # 必开
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=16, pin_memory=True, persistent_workers=True)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=512, shuffle=False, 
                            num_workers=16, pin_memory=True)

    # --- 模型实例化 ---
    net_client1 = NetClient1(in_=3, num_class=NUM_CLASSES, split_block=SPLIT_BLOCK).to(DEVICE)
    net_server = NetServer(num_class=NUM_CLASSES, split_block=SPLIT_BLOCK).to(DEVICE)
    net_client2 = NetClient2(num_class=NUM_CLASSES).to(DEVICE)

    # --- 关键：权重初始化 ---
    print("🎨 Applying Kaiming Initialization...")
    net_client1.apply(weights_init)
    net_server.apply(weights_init)
    # NetClient2 内部调用了 torchvision 的 vgg classifier，最好也重置一下以防万一
    net_client2.apply(weights_init)

    # --- 编译加速 ---
    print("⚡ Compiling models (mode='default')...")
    net_client1 = torch.compile(net_client1, mode='default')
    net_server = torch.compile(net_server, mode='default')
    net_client2 = torch.compile(net_client2, mode='default')

    # --- 优化器选择 (SGD vs AdamW) ---
    # VGG 是无残差深层网络，SGD+Momentum 通常比 AdamW 更容易获得高泛化精度
    params = list(net_client1.parameters()) + list(net_server.parameters()) + list(net_client2.parameters())
    optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)

    # --- 学习率调度 (Warmup + Cosine) ---
    # VGG 训练初期非常不稳定，Warmup 是必须的
    # 我们使用 OneCycleLR 或者 组合调度，这里用 Cosine 配合前几轮手动 Warmup 逻辑更直观
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- Loss ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        # --- Warmup 策略 (前5轮线性增加 LR) ---
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

            # --- 1. Mixup (准确率提升秘籍) ---
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, device=DEVICE)

            # --- 2. Forward (BF16 AMP) ---
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                client1_out = net_client1(inputs)
                server_out = net_server(client1_out)
                outputs = net_client2(server_out)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # --- 3. Backward (Split Learning) ---
            optimizer.zero_grad()
            loss.backward()

            # 手动反向传播
            grad_to_server = net_client2.client_backward()
            grad_to_client1 = net_server.server_backward(grad_to_server)
            net_client1.client_backward(grad_to_client1)

            # --- 4. 梯度裁剪 (防止 VGG 梯度爆炸) ---
            # 裁剪所有子模型的梯度范数
            nn.utils.clip_grad_norm_(net_client1.parameters(), max_norm=5.0)
            nn.utils.clip_grad_norm_(net_server.parameters(), max_norm=5.0)
            nn.utils.clip_grad_norm_(net_client2.parameters(), max_norm=5.0)

            optimizer.step()

            # --- 统计 ---
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # Mixup 下的准确率估算
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            
            pbar.set_postfix({
                "Loss": f"{running_loss/(total/BATCH_SIZE):.3f}", 
                "Acc": f"{100.*correct/total:.2f}%", 
                "LR": f"{optimizer.param_groups[0]['lr']:.4f}"
            })

        # Epoch 结束，更新学习率 (Warmup 结束后)
        if epoch >= 5:
            scheduler.step()

        # --- 验证 ---
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

        if acc > best_acc:
            best_acc = acc
            tqdm.write(f"🏆 New Record! Acc: {best_acc:.2f}%")
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