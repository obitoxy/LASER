"""
lib.py - Shared framework for the LASER *scale* experiment (RAID 2026 rebuttal
to reviewers 296A #4 / 296B #5: "larger tasks").

It is a direct adaptation of exp_robust/lib.py to CIFAR-100 (100 classes), with
the single mandatory code change the scale experiment requires: the silhouette
K-search upper bound is raised from max_k=12 (a CIFAR-10 / 10-class ceiling) to
max_k=120, so the search can actually discover K* in the ~100 range. With the
old ceiling the search would silently saturate at K*=12 and the experiment would
fail *invisibly*.

The pipeline is otherwise IDENTICAL to exp_robust and reuses the same src/
building blocks (VGG16/SB-1 model factory, MyDataLoader, attacks_dataloader1,
find_optimal_k_conservative), so the 10-class -> 100-class comparison changes
exactly one variable: the label-space size K. Architecture (VGG16), split point
(SB=1), input resolution (32x32), and the LASER mechanism are all held fixed,
per the control-variable rationale for the rebuttal.

Pipeline:
    train VGG16/SB-1 on CIFAR-100 -> cache server embeddings -> unsupervised
    clustering (silhouette K search, max_k=120) -> LASER attack on a
    label-free (blind) sample of clusters -> evaluation.

Disk safety: the ~10 GB embedding cache is freed as soon as clustering + attacks
+ evaluation finish (see exp_scale/main.py), exactly as in exp_robust.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")           # avoid invalid-value warn
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score)

from src.models.vgg16 import NetClient1, NetServer, NetClient2
from src.MyDataLoader import MyDataLoader
from src.attack import attacks_dataloader1
from src.embedding_clustering import find_optimal_k_conservative

# ----------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "exp_scale", "outputs")
os.makedirs(OUT, exist_ok=True)

# CIFAR-100 normalization (standard CIFAR-100 stats; differ slightly from CIFAR-10).
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)

# CIFAR-100 fine-label class names (torchvision order, idx 0..99).
CLASS_NAMES = (
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
)
assert len(CLASS_NAMES) == 100

SPLIT_BLOCK = 1          # SB=1 (Shallow split) - same as every §6 rebuttal run
NUM_CLASSES = 100        # <-- the one variable that changes vs exp_robust
MAX_K = 120              # <-- mandatory fix: K-search ceiling for 100 classes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ======================================================================
# 1. DATA (CIFAR-100)
# ======================================================================
def _base_cifar100(root='./data'):
    """Raw CIFAR-100 train/test tensors + integer fine labels (no transform)."""
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=False)
    test = torchvision.datasets.CIFAR100(root=root, train=False, download=False)
    xtr = train.data                      # (50000, 32, 32, 3) uint8
    ytr = np.array(train.targets)
    xte = test.data
    yte = np.array(test.targets)
    return xtr, ytr, xte, yte


class ImageArrayDataset(torchvision.datasets.VisionDataset):
    """Array-backed image dataset (uint8 HWC + integer labels) + torchvision
    transforms. Dataset-agnostic generalization of exp_robust.CIFAR10Array."""
    def __init__(self, images, labels, transform=None):
        super().__init__(root=None, transform=transform)
        self.images = np.ascontiguousarray(images, dtype=np.uint8)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def make_loaders(train_images, train_labels, batch_size=512, num_workers=8):
    """Build train (augmented) + test (clean, standard CIFAR-100) loaders.
    The TEST set is always the standard clean balanced CIFAR-100 test set so
    ACC-V / NVA Drop / MP are measured against ground truth."""
    trainset = ImageArrayDataset(train_images, train_labels, transform=get_transforms(True))
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False,
                                            transform=get_transforms(False))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=512, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return trainloader, testloader, trainset, testset


# ======================================================================
# 2. TRAINING  (identical protocol to exp_robust: SGD+momentum, mixup, cosine)
# ======================================================================
def kaiming_init(m):
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


def _mixup(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def create_models(sb=SPLIT_BLOCK):
    net1 = NetClient1(in_=3, num_class=NUM_CLASSES, split_block=sb).to(DEVICE)
    net2 = NetServer(num_class=NUM_CLASSES, split_block=sb).to(DEVICE)
    net3 = NetClient2(num_class=NUM_CLASSES).to(DEVICE)
    for m in (net1, net2, net3):
        m.apply(kaiming_init)
    return net1, net2, net3


@torch.no_grad()
def evaluate_acc(net1, net2, net3, testloader):
    net1.eval(); net2.eval(); net3.eval()
    correct = total = 0
    per_correct = np.zeros(NUM_CLASSES)
    per_total = np.zeros(NUM_CLASSES)
    for x, y in testloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = net3(net2(net1(x)))
        pred = out.argmax(1)
        correct += pred.eq(y).sum().item(); total += y.size(0)
        for c in range(NUM_CLASSES):
            m = y == c
            per_total[c] += m.sum().item()
            per_correct[c] += (pred[m] == c).sum().item()
    acc = 100.0 * correct / total
    per_class = 100.0 * per_correct / np.maximum(per_total, 1)
    return acc, per_class


def train_split_model(tag, train_images, train_labels, epochs=120,
                      lr=0.05, wd=5e-4, batch_size=512):
    """Train a USL split model (VGG16, SB=1) on CIFAR-100. Same optimizer /
    schedule / augmentation as the exp_robust baseline so the only difference
    vs the CIFAR-10 control is the number of classes. Saves best + nor_last."""
    save_dir = os.path.join(OUT, "models", tag)
    os.makedirs(save_dir, exist_ok=True)
    trainloader, testloader, _, _ = make_loaders(train_images, train_labels, batch_size)

    net1, net2, net3 = create_models()
    params = list(net1.parameters()) + list(net2.parameters()) + list(net3.parameters())
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    history = []
    t_start = time.time()
    for epoch in range(epochs):
        if epoch < 5:  # warmup
            wl = lr * (epoch + 1) / 5
            for g in optimizer.param_groups:
                g['lr'] = wl
        net1.train(); net2.train(); net3.train()
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x, ya, yb, lam = _mixup(x, y, 1.0)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                c2s = net1(x); s2c = net2(c2s); out = net3(s2c)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
            loss.backward()
            g2s = net3.client_backward(); g2c = net2.server_backward(g2s); net1.client_backward(g2c)
            for mm in (net1, net2, net3):
                nn.utils.clip_grad_norm_(mm.parameters(), 5.0)
            optimizer.step()
        if epoch >= 5:
            scheduler.step()

        acc, per_class = evaluate_acc(net1, net2, net3, testloader)
        history.append({"epoch": epoch + 1, "acc": acc})
        elapsed = time.time() - t_start
        print(f"[{tag}] epoch {epoch+1}/{epochs}  test_acc={acc:.2f}%  "
              f"best={max(best_acc,acc):.2f}%  ({elapsed/60:.1f}m)", flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save(net1.state_dict(), os.path.join(save_dir, "net1_best.pt"))
            torch.save(net2.state_dict(), os.path.join(save_dir, "server_best.pt"))
            torch.save(net3.state_dict(), os.path.join(save_dir, "net3_best.pt"))

    torch.save(net1.state_dict(), os.path.join(save_dir, "net1_nor_last.pt"))
    torch.save(net2.state_dict(), os.path.join(save_dir, "server_nor_last.pt"))
    torch.save(net3.state_dict(), os.path.join(save_dir, "net3_nor_last.pt"))
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump({"best_acc": best_acc, "history": history}, f, indent=2)
    total_min = (time.time() - t_start) / 60.0
    print(f"[{tag}] training DONE  best_acc={best_acc:.2f}%  "
          f"total {total_min:.1f}m", flush=True)
    return save_dir, best_acc


# ======================================================================
# 3. EMBEDDING COLLECTION  (identical to exp_robust; dataset-agnostic)
# ======================================================================
@torch.no_grad()
def collect_embeddings(tag, train_images, train_labels_clean, model_dir,
                       batch_size=256):
    """Forward the (clean-label) CIFAR-100 training images through the trained
    split model to cache smashed data (c2s) and server output (s2c).
    labels_true is CLEAN ground truth, used ONLY for cluster-quality measurement,
    never by the (label-free) attack. Returns memmap-backed np arrays."""
    net1, net2, net3 = create_models()
    net1.load_state_dict(torch.load(os.path.join(model_dir, "net1_nor_last.pt")))
    net2.load_state_dict(torch.load(os.path.join(model_dir, "server_nor_last.pt")))
    net3.load_state_dict(torch.load(os.path.join(model_dir, "net3_nor_last.pt")))
    net1.eval(); net2.eval(); net3.eval()

    ds = ImageArrayDataset(train_images, train_labels_clean, transform=get_transforms(False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)

    cache_dir = os.path.join(OUT, "cache", tag)
    os.makedirs(cache_dir, exist_ok=True)
    c2s_path = os.path.join(cache_dir, "embeds_c2s.npy")
    s2c_path = os.path.join(cache_dir, "embeds_s2c.npy")
    lab_path = os.path.join(cache_dir, "labels_true.npy")

    n = len(ds)
    fp_c2s = fp_s2c = fp_lab = None
    s = 0
    for x, y in loader:
        x = x.to(DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            c2s = net1(x).float()
            s2c = net2(c2s).float()
        c2s = c2s.cpu().numpy().astype(np.float32)
        s2c = s2c.cpu().numpy().astype(np.float32)
        if fp_c2s is None:
            fp_c2s = np.lib.format.open_memmap(c2s_path, 'w+', np.float32,
                                               (n,) + c2s.shape[1:])
            fp_s2c = np.lib.format.open_memmap(s2c_path, 'w+', np.float32,
                                               (n,) + s2c.shape[1:])
            fp_lab = np.lib.format.open_memmap(lab_path, 'w+', np.int64, (n,))
        e = s + x.size(0)
        fp_c2s[s:e] = c2s; fp_s2c[s:e] = s2c; fp_lab[s:e] = y.numpy()
        s = e
    fp_c2s.flush(); fp_s2c.flush(); fp_lab.flush()
    print(f"[{tag}] cached embeddings: c2s {fp_c2s.shape}, s2c {fp_s2c.shape}")
    return c2s_path, s2c_path, lab_path


# ======================================================================
# 4. UNSUPERVISED CLUSTERING (silhouette K search, max_k=120)
# ======================================================================
def cluster_features(tag, s2c_path, lab_path, max_k=MAX_K, sample=4000,
                     pca_var=0.99, pca_cap=256, seed=42):
    """Run the paper's silhouette-based K search, then final K-means clustering
    on the server output features.

    IDENTICAL procedure to exp_robust.lib.cluster_features except max_k defaults
    to 120 (not 12) so the search can reach the ~100-class regime, and the
    purity bincount uses NUM_CLASSES=100. PCA-reduces the 512*7*7=25,088-dim
    server features (fit on a subsample, transform all) and performs K-search +
    final clustering in that space; PCA features/centers are cached for the
    (label-free) attack's priority sampler. The attack's own centroid/MSE still
    operates in the native (512,7,7) space."""
    s2c = np.load(s2c_path, mmap_mode='r')
    labels_true = np.load(lab_path)
    n = s2c.shape[0]
    flat = np.asarray(s2c.reshape(n, -1), dtype=np.float32)   # zero-copy view

    # --- PCA (fit on subsample, transform all) ---
    rng = np.random.RandomState(seed)
    sub_idx = rng.choice(n, min(sample, n), replace=False)
    sub = flat[sub_idx]
    t0 = time.time()
    pca = PCA(n_components=pca_var, random_state=seed).fit(sub)
    if pca.n_components_ > pca_cap:
        pca = PCA(n_components=pca_cap, random_state=seed).fit(sub)
    feat = pca.transform(flat).astype(np.float32)             # n x d
    print(f"[{tag}] PCA -> {pca.n_components_} dims "
          f"(kept {float(np.sum(pca.explained_variance_ratio_)):.3f} var) "
          f"in {time.time()-t0:.1f}s")

    # --- K search (silhouette, conservative) on PCA subsample ---
    sub_p = feat[sub_idx]
    k_star = find_optimal_k_conservative(sub_p, max_k=max_k)
    sil_curve = {}
    t0 = time.time()
    for k in range(2, max_k + 1):
        kk = KMeans(n_clusters=k, random_state=seed, n_init=3).fit_predict(sub_p)
        sil_curve[k] = float(silhouette_score(sub_p, kk))
    print(f"[{tag}] silhouette curve (k=2..{max_k}) computed in "
          f"{time.time()-t0:.1f}s; argmax k={max(sil_curve, key=sil_curve.get)} "
          f"(sil={max(sil_curve.values()):.4f})")

    # --- final K-means at k* on ALL PCA-reduced features ---
    km = KMeans(n_clusters=k_star, random_state=seed, n_init=10)
    clabels = km.fit_predict(feat)
    centers = km.cluster_centers_.astype(np.float32)          # PCA space

    # --- clustering quality vs ground-truth CLEAN labels ---
    ari = float(adjusted_rand_score(labels_true, clabels))
    nmi = float(normalized_mutual_info_score(labels_true, clabels))
    purity = 0.0
    for c in range(k_star):
        m = clabels == c
        if m.sum() == 0:
            continue
        purity += np.bincount(labels_true[m], minlength=NUM_CLASSES).max()
    purity /= n

    cluster_to_class = {}
    per_cluster_purity = {}
    for c in range(k_star):
        m = clabels == c
        if m.sum() == 0:
            continue
        cnt = np.bincount(labels_true[m], minlength=NUM_CLASSES)
        maj = int(cnt.argmax())
        cluster_to_class[c] = maj
        per_cluster_purity[c] = (float(cnt.max() / m.sum()), int(m.sum()), maj)

    info = {
        "tag": tag, "n": int(n), "pca_dim": int(pca.n_components_),
        "max_k": int(max_k), "k_star": int(k_star), "ari": ari, "nmi": nmi,
        "purity": purity,
        "sil_curve": {int(k): v for k, v in sil_curve.items()},
        "per_cluster": {int(c): {"purity": p[0], "size": p[1], "maj_class": p[2]}
                        for c, p in per_cluster_purity.items()},
        "cluster_to_class": {int(c): int(v) for c, v in cluster_to_class.items()},
    }
    cluster_dir = os.path.join(OUT, "clusters", tag)
    os.makedirs(cluster_dir, exist_ok=True)
    np.save(os.path.join(cluster_dir, "cluster_labels.npy"), clabels)
    feat_path = os.path.join(cluster_dir, "feat_pca.npy")
    np.save(feat_path, feat)
    np.save(os.path.join(cluster_dir, "centers_pca.npy"), centers)
    with open(os.path.join(cluster_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[{tag}] cluster: k*={k_star} (max_k={max_k}) ARI={ari:.3f} "
          f"NMI={nmi:.3f} purity={purity:.3f}")
    return clabels, centers, feat_path, info


# ======================================================================
# 5. LASER ATTACK  (reuses the paper's attacks_dataloader1, unchanged)
# ======================================================================
def run_laser_attack(tag, c2s_path, s2c_path, feat_path, centers_pca, cluster_labels,
                     target_cluster, proxy_cluster, atk_epochs=20,
                     count=10, count2=50, alpha=0.5, lr=1e-4, suffix="",
                     model_tag=None):
    """Run LASER (VGG last-conv-block fine-tune, guided-centroid pull + anchor).

    Identical to exp_robust.lib.run_laser_attack; the attack operates in native
    server-output space (512,7,7) and the cluster partition comes from PCA-space
    clustering. With 100 clusters the per-attack cached budget is
    count=10 victim + count2=50 x 99 non-victim = 4,960 vectors (still << 50,000
    cache), as anticipated in the rebuttal plan."""
    embeds_c2s = np.load(c2s_path, mmap_mode='r')
    embeds_s2c = np.load(s2c_path, mmap_mode='r')
    labels = cluster_labels.astype(np.int64)

    # priority-sampler distances in (cheap) PCA space
    feat = np.load(feat_path, mmap_mode='r')
    distances = np.linalg.norm(
        np.asarray(feat, dtype=np.float32) - centers_pca[labels], axis=1
    ).astype(np.float32)

    net1, server, net3 = create_models()
    server.load_state_dict(torch.load(
        os.path.join(OUT, "models", model_tag or tag, "server_nor_last.pt")))

    dataloader = MyDataLoader(
        embeds_c2s=embeds_c2s, embeds_s2c=embeds_s2c, labels=labels,
        batch_size=50, distances=distances, shuffle=True, drop_last=False,
        count=count, count2=count2, target_label=target_cluster)

    save_dir = os.path.join(OUT, "attacks", f"{tag}_t{target_cluster}_p{proxy_cluster}{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    attacks_dataloader1(dataloader, server, target_cluster, proxy_cluster,
                        DEVICE, atk_epochs, lr, save_dir, np.asarray(embeds_s2c), labels)
    return save_dir


# ======================================================================
# 6. EVALUATION
# ======================================================================
@torch.no_grad()
def evaluate_attack(tag, model_dir, atk_dir, atk_epoch,
                    victim_class, proxy_class):
    """Measure ACC-V, NVA Drop, MP and per-class accuracy on the clean balanced
    CIFAR-100 test set using: clean net1 + clean net3 + ATTACKED server."""
    net1, server, net3 = create_models()
    net1.load_state_dict(torch.load(os.path.join(model_dir, "net1_nor_last.pt")))
    server.load_state_dict(torch.load(os.path.join(atk_dir, f"server_{atk_epoch}.pt")))
    net3.load_state_dict(torch.load(os.path.join(model_dir, "net3_nor_last.pt")))
    net1.eval(); server.eval(); net3.eval()

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False,
                                            transform=get_transforms(False))
    loader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=8)

    per_correct = np.zeros(NUM_CLASSES)
    per_total = np.zeros(NUM_CLASSES)
    victim_total = 0
    victim_to_proxy = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = net3(server(net1(x)))
        pred = out.argmax(1)
        for c in range(NUM_CLASSES):
            m = y == c
            per_total[c] += m.sum().item()
            per_correct[c] += (pred[m] == c).sum().item()
        mv = y == victim_class
        victim_total += mv.sum().item()
        victim_to_proxy += (pred[mv] == proxy_class).sum().item()
    per_class = 100.0 * per_correct / np.maximum(per_total, 1)

    acc_v = per_class[victim_class] if per_total[victim_class] > 0 else 0.0
    mp = 100.0 * victim_to_proxy / max(victim_total, 1)
    return per_class, acc_v, mp


@torch.no_grad()
def preattack_per_class(model_dir):
    """Pre-attack (clean server) per-class accuracy on CIFAR-100 - for NVA Drop."""
    net1, server, net3 = create_models()
    net1.load_state_dict(torch.load(os.path.join(model_dir, "net1_nor_last.pt")))
    server.load_state_dict(torch.load(os.path.join(model_dir, "server_nor_last.pt")))
    net3.load_state_dict(torch.load(os.path.join(model_dir, "net3_nor_last.pt")))
    net1.eval(); server.eval(); net3.eval()
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False,
                                            transform=get_transforms(False))
    loader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=8)
    per_correct = np.zeros(NUM_CLASSES); per_total = np.zeros(NUM_CLASSES)
    overall_c = overall_t = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = net3(server(net1(x)))
        pred = out.argmax(1)
        overall_c += pred.eq(y).sum().item(); overall_t += y.size(0)
        for c in range(NUM_CLASSES):
            m = y == c
            per_total[c] += m.sum().item()
            per_correct[c] += (pred[m] == c).sum().item()
    return 100.0 * per_correct / np.maximum(per_total, 1), 100.0 * overall_c / overall_t


def nva_drop(per_class_pre, per_class_post, victim_class):
    """NVA Drop = mean non-victim accuracy (pre) - (post), averaged over the
    NUM_CLASSES-1 non-victim classes (matches the paper's definition)."""
    others = [c for c in range(NUM_CLASSES) if c != victim_class]
    pre = np.mean([per_class_pre[c] for c in others])
    post = np.mean([per_class_post[c] for c in others])
    return float(pre - post), float(pre), float(post)


# ======================================================================
# Convenience helpers (mirrors exp_robust)
# ======================================================================
def blob_cluster_ids(info, factor=2.0):
    """Label-free detection of over-clustering 'garbage blob' clusters.

    When K-means runs with K slightly above the true class count (here K*=102 vs
    100 true classes) the extra capacity is spent on a single large central
    cluster that absorbs ambiguous/hard samples - a blended blob whose centroid
    sits near the global mean. Such a blob is NOT a separable semantic cluster
    (the only thing the §6.3 threat model targets), and its central centroid
    fools the nearest-centroid proxy heuristic (it is 'nearest' to many clusters).

    A blind attacker can detect these blobs WITHOUT labels: they are anomalously
    large (size an outlier vs the rest). We flag any cluster whose size exceeds
    `factor` x the median cluster size. Returns the set of blob cluster ids."""
    sizes = np.array([pc["size"] for pc in info["per_cluster"].values()])
    if len(sizes) == 0:
        return set()
    thr = factor * float(np.median(sizes))
    blobs = {int(c) for c, pc in info["per_cluster"].items()
             if pc["size"] > thr}
    return blobs


def nearest_other_cluster(centers, tc, forbidden_class=None, cluster_to_class=None,
                          forbidden_clusters=None):
    """Label-free proxy heuristic: the cluster (!= tc) whose PCA centroid is
    nearest to cluster tc's centroid. If forbidden_class + cluster_to_class are
    given, also skip clusters whose majority class equals the victim's so the
    proxy is a *different* semantic region (labels used only to avoid a
    degenerate same-class redirect; the geometric ranking is label-free). If
    `forbidden_clusters` (a set of cluster ids, e.g. the garbage blobs) is given,
    skip those too - a label-free guard that prevents the proxy from being a
    central blended blob (see blob_cluster_ids)."""
    d = np.linalg.norm(centers - centers[tc], axis=1).astype(np.float64)
    d[tc] = np.inf
    if forbidden_clusters:
        for c in forbidden_clusters:
            if 0 <= c < len(d):
                d[c] = np.inf
    if forbidden_class is not None and cluster_to_class is not None:
        for c in range(len(centers)):
            if c == tc:
                continue
            if int(cluster_to_class.get(str(c), -1)) == forbidden_class:
                d[c] = np.inf
    pc = int(np.argmin(d))
    if not np.isfinite(d[pc]):
        raise ValueError(f"no valid proxy cluster for victim cluster {tc}")
    return pc
