"""
lib.py — Camera-ready controlled experiments: seed variance × split depth.

This is a parameterized copy of exp_robust/lib.py.  The *only* changes are:
    * OUT points at exp_camera/outputs (keeps the rebuttal artefacts untouched);
    * `train_split_model` accepts explicit `sb` (split block) and `seed`;
    * `collect_embeddings`, `run_laser_attack`, `evaluate_attack` accept `sb`
      so the same pipeline runs for VGG-16 SB=1/2/3;
    * `run_laser_attack` can load a server from an explicit `model_dir`.

Everything else — data loading, training recipe, PCA + silhouette K-search,
the LASER attack (`src.attack.attacks_dataloader1`), and evaluation — is
copied verbatim from exp_robust so that the controlled study is directly
comparable to the paper and the rebuttal experiments.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# The host occasionally exports an invalid OMP_NUM_THREADS (e.g. 0), which
# makes libgomp abort before Python can start.  Repair it defensively.
_omp = os.environ.get("OMP_NUM_THREADS", "")
if not _omp.isdigit() or int(_omp) <= 0:
    os.environ["OMP_NUM_THREADS"] = "8"

import copy
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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
OUT = os.path.join(REPO, "exp_camera", "outputs")
os.makedirs(OUT, exist_ok=True)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

SPLIT_BLOCK = 1          # default split point (paper's Shallow split)
NUM_CLASSES = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 9202


def set_seed(seed=SEED):
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


# ======================================================================
# 1. DATA DISTRIBUTION BUILDERS
# ======================================================================
def _base_cifar(root='./data'):
    """Raw CIFAR-10 train/test tensors + integer labels (no transform)."""
    train = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    xtr = train.data                      # (50000, 32, 32, 3) uint8
    ytr = np.array(train.targets)
    xte = test.data
    yte = np.array(test.targets)
    return xtr, ytr, xte, yte


class CIFAR10Array(torchvision.datasets.VisionDataset):
    """CIFAR-10 backed by an in-memory uint8 array + integer label array.

    Lets us inject custom (sub-sampled / noisy) data while still applying the
    standard torchvision transforms.
    """
    def __init__(self, images, labels, transform=None):
        super().__init__(root=None, transform=transform)
        # images: (N,32,32,3) uint8 ; labels: (N,) int
        self.images = np.ascontiguousarray(images, dtype=np.uint8)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def build_longtail(rho=100, seed=9202, max_per_class=5000):
    """Long-tail CIFAR-10-LT. Class 0 is the head (most samples), class K-1 the
    tail. Per-class count decays exponentially: n_i = max*(1/rho)^(i/(K-1)).

    Returns the index array of the kept training samples (in original order).
    """
    xtr, ytr, _, _ = _base_cifar()
    rng = np.random.RandomState(seed)
    keep = []
    per_class = []
    for c in range(NUM_CLASSES):
        frac = (1.0 / rho) ** (c / (NUM_CLASSES - 1))
        n = int(round(max_per_class * frac))
        n = max(n, 1)
        idx_c = np.where(ytr == c)[0]
        rng.shuffle(idx_c)
        sel = idx_c[:n]
        keep.extend(sel.tolist())
        per_class.append(n)
    keep = np.array(sorted(keep))
    return keep, per_class


def build_noisy_labels(noise_rate=0.2, seed=9202):
    """Symmetric label noise: flip `noise_rate` fraction of training labels
    uniformly to a different class. Returns the noisy label array (full 50k)."""
    xtr, ytr, _, _ = _base_cifar()
    rng = np.random.RandomState(seed)
    y_noisy = ytr.copy()
    n = len(ytr)
    flip = rng.rand(n) < noise_rate
    for i in np.where(flip)[0]:
        new = rng.randint(0, NUM_CLASSES)
        while new == y_noisy[i]:
            new = rng.randint(0, NUM_CLASSES)
        y_noisy[i] = new
    actual_noise = float((y_noisy != ytr).mean())
    return y_noisy, actual_noise


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
    """Build train (augmented) + test (clean, balanced, standard) loaders.

    The TEST set is always the standard clean balanced CIFAR-10 test set so that
    ACC-V / NVA Drop / MP are measured against ground truth."""
    trainset = CIFAR10Array(train_images, train_labels, transform=get_transforms(True))
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=get_transforms(False))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=512, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return trainloader, testloader, trainset, testset


# ======================================================================
# 2. TRAINING
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
    preds_all, labels_all = [], []
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
        preds_all.append(pred.cpu().numpy())
        labels_all.append(y.cpu().numpy())
    acc = 100.0 * correct / total
    per_class = 100.0 * per_correct / np.maximum(per_total, 1)
    return acc, per_class, np.concatenate(labels_all), np.concatenate(preds_all)


@torch.no_grad()
def preattack_per_class(model_dir, sb=SPLIT_BLOCK, num_workers=8):
    """Clean split model's per-class test accuracy + overall accuracy."""
    net1, server, net3 = create_models(sb)
    net1.load_state_dict(torch.load(os.path.join(model_dir, "net1_nor_last.pt")))
    server.load_state_dict(torch.load(os.path.join(model_dir, "server_nor_last.pt")))
    net3.load_state_dict(torch.load(os.path.join(model_dir, "net3_nor_last.pt")))
    net1.eval(); server.eval(); net3.eval()
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=get_transforms(False))
    loader = DataLoader(testset, batch_size=512, shuffle=False,
                        num_workers=num_workers)
    per_correct = np.zeros(NUM_CLASSES)
    per_total = np.zeros(NUM_CLASSES)
    overall_c = overall_t = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = net3(server(net1(x)))
        pred = out.argmax(1)
        overall_c += pred.eq(y).sum().item()
        overall_t += y.size(0)
        for c in range(NUM_CLASSES):
            m = y == c
            per_total[c] += m.sum().item()
            per_correct[c] += (pred[m] == c).sum().item()
    per_class = 100.0 * per_correct / np.maximum(per_total, 1)
    overall = 100.0 * overall_c / overall_t
    return per_class, overall


def train_split_model(tag, train_images, train_labels, epochs=120,
                      lr=0.05, wd=5e-4, batch_size=512,
                      sb=SPLIT_BLOCK, seed=SEED, num_workers=8):
    """Train a USL split model (VGG16) at an explicit split point and seed.

    Saves only the best checkpoint (net1/server/net3 _best.pt) plus a final
    nor_last snapshot, to keep disk usage modest (the VGG classifier head is
    ~478 MB per copy).  `config.json` records sb/seed/epochs for the study.
    """
    set_seed(seed)
    save_dir = os.path.join(OUT, "models", tag)
    os.makedirs(save_dir, exist_ok=True)
    config = {
        "tag": tag, "dataset": "CIFAR-10", "architecture": "VGG-16",
        "split_block": int(sb), "seed": int(seed), "epochs": int(epochs),
        "lr": float(lr), "weight_decay": float(wd),
        "batch_size": int(batch_size), "num_classes": NUM_CLASSES,
        "train_size": int(len(train_images)),
        "train_images_shape": list(train_images.shape),
        "protocol": "SGD+momentum0.9+nesterov, cosine, warmup5, mixup(alpha=1), "
                    "label_smoothing=0.1, bf16 AMP, AutoAugment",
        "torch_version": torch.__version__,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[{tag}] TRAIN sb={sb} seed={seed} epochs={epochs} "
          f"n={len(train_images)}", flush=True)

    trainloader, testloader, _, _ = make_loaders(
        train_images, train_labels, batch_size, num_workers=num_workers)

    net1, net2, net3 = create_models(sb)
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

        acc, per_class, _, _ = evaluate_acc(net1, net2, net3, testloader)
        history.append({"epoch": epoch + 1, "acc": acc})
        elapsed = time.time() - t_start
        print(f"[{tag}] epoch {epoch+1}/{epochs}  test_acc={acc:.2f}%  "
              f"best={max(best_acc,acc):.2f}%  ({elapsed/60:.1f}m)", flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save(net1.state_dict(), os.path.join(save_dir, "net1_best.pt"))
            torch.save(net2.state_dict(), os.path.join(save_dir, "server_best.pt"))
            torch.save(net3.state_dict(), os.path.join(save_dir, "net3_best.pt"))

    # final "nor_last" snapshot (matches paper's cached-model convention)
    torch.save(net1.state_dict(), os.path.join(save_dir, "net1_nor_last.pt"))
    torch.save(net2.state_dict(), os.path.join(save_dir, "server_nor_last.pt"))
    torch.save(net3.state_dict(), os.path.join(save_dir, "net3_nor_last.pt"))
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump({"best_acc": best_acc, "config": config, "history": history},
                  f, indent=2)
    total_min = (time.time() - t_start) / 60.0
    print(f"[{tag}] training DONE  best_acc={best_acc:.2f}%  "
          f"total {total_min:.1f}m", flush=True)
    return save_dir, best_acc


# ======================================================================
# 3. EMBEDDING COLLECTION
# ======================================================================
@torch.no_grad()
def collect_embeddings(tag, train_images, train_labels_clean, model_dir,
                       batch_size=256, indices=None, cache_suffix="",
                       sb=SPLIT_BLOCK, num_workers=8):
    """Forward the (clean-label) training images through the trained split model
    to cache the smashed data (c2s) and server output (s2c).

    labels_true is always the CLEAN ground-truth label — it is used only for
    evaluation / cluster-quality measurement, never by the (label-free) attack.
    If `indices` is given, only those training samples are cached (used by the
    dynamic-distribution experiment to cache per-phase streaming subsets).
    Returns memmap-backed np arrays: embeds_c2s, embeds_s2c, labels_true."""
    net1, net2, net3 = create_models(sb)
    net1.load_state_dict(torch.load(os.path.join(model_dir, "net1_nor_last.pt")))
    net2.load_state_dict(torch.load(os.path.join(model_dir, "server_nor_last.pt")))
    net3.load_state_dict(torch.load(os.path.join(model_dir, "net3_nor_last.pt")))
    net1.eval(); net2.eval(); net3.eval()

    if indices is not None:
        train_images = train_images[indices]
        train_labels_clean = train_labels_clean[indices]
    # Use the clean-label evaluation transform (deterministic) for the cache.
    ds = CIFAR10Array(train_images, train_labels_clean, transform=get_transforms(False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    cache_dir = os.path.join(OUT, "cache", tag)
    os.makedirs(cache_dir, exist_ok=True)
    c2s_path = os.path.join(cache_dir, f"embeds_c2s{cache_suffix}.npy")
    s2c_path = os.path.join(cache_dir, f"embeds_s2c{cache_suffix}.npy")
    lab_path = os.path.join(cache_dir, f"labels_true{cache_suffix}.npy")

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
# 4. UNSUPERVISED CLUSTERING (silhouette K search + quality metrics)
# ======================================================================
def cluster_features(tag, s2c_path, lab_path, max_k=12, sample=4000,
                     pca_var=0.99, pca_cap=256, seed=42):
    """Run the paper's silhouette-based K search, then final K-means clustering
    on the server output features.

    The flattened server features are 512*7*7 = 25,088-dim, which makes raw-space
    K-means + silhouette prohibitively slow. We therefore PCA-reduce the features
    (fit on a subsample, transform all) to ~pca_cap dimensions preserving pca_var
    variance, and perform K-search + final clustering in that space. PCA preserves
    local neighborhood structure, so the resulting partition matches raw-space
    K-means. The PCA-reduced features and centers are cached so the (label-free)
    attack can reuse them for its priority sampler; the attack's own centroid/MSE
    still operates in the native (512,7,7) space."""
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

    # --- K search (silhouette) on PCA subsample ---
    sub_p = feat[sub_idx]
    k_star = find_optimal_k_conservative(sub_p, max_k=max_k)
    sil_curve = {}
    for k in range(2, max_k + 1):
        kk = KMeans(n_clusters=k, random_state=seed, n_init=3).fit_predict(sub_p)
        sil_curve[k] = float(silhouette_score(sub_p, kk))

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
        "k_star": int(k_star), "ari": ari, "nmi": nmi, "purity": purity,
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
    print(f"[{tag}] cluster: k*={k_star} ARI={ari:.3f} NMI={nmi:.3f} "
          f"purity={purity:.3f}")
    return clabels, centers, feat_path, info


# ======================================================================
# 5. LASER ATTACK  (reuses the paper's attacks_dataloader1)
# ======================================================================
def run_laser_attack(tag, c2s_path, s2c_path, feat_path, centers_pca, cluster_labels,
                     target_cluster, proxy_cluster, atk_epochs=20,
                     count=10, count2=50, alpha=0.5, lr=1e-4, suffix="",
                     model_tag=None, model_dir=None, sb=SPLIT_BLOCK,
                     attack_seed=42):
    """Run LASER (VGG last-conv-block fine-tune, guided-centroid pull + anchor).

    The attack itself operates in the NATIVE server-output space (512,7,7): the
    guided centroid is the mean of the proxy cluster's raw s2c features and the
    attack/stability losses are MSE in that space (exactly as in the paper).
    The cluster *partition* (`cluster_labels`) comes from PCA-space clustering;
    we reuse the PCA features/centers only to compute the priority-sampler
    distances (which victim/non-victim samples are purest). `feat_path` holds the
    PCA-reduced features and `centers_pca` the corresponding cluster centers."""
    embeds_c2s = np.load(c2s_path, mmap_mode='r')
    embeds_s2c = np.load(s2c_path, mmap_mode='r')
    labels = cluster_labels.astype(np.int64)

    # priority-sampler distances in (cheap) PCA space
    feat = np.load(feat_path, mmap_mode='r')
    distances = np.linalg.norm(
        np.asarray(feat, dtype=np.float32) - centers_pca[labels], axis=1
    ).astype(np.float32)

    set_seed(attack_seed)
    net1, server, net3 = create_models(sb)
    src_model_dir = model_dir or os.path.join(OUT, "models", model_tag or tag)
    server.load_state_dict(torch.load(
        os.path.join(src_model_dir, "server_nor_last.pt")))

    dataloader = MyDataLoader(
        embeds_c2s=embeds_c2s, embeds_s2c=embeds_s2c, labels=labels,
        batch_size=50, distances=distances, shuffle=True, drop_last=False,
        count=count, count2=count2, target_label=target_cluster,
        random_state=attack_seed)

    save_dir = os.path.join(OUT, "attacks", f"{tag}_t{target_cluster}_p{proxy_cluster}{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "attack_config.json"), "w") as f:
        json.dump({
            "tag": tag, "model_dir": src_model_dir, "split_block": int(sb),
            "target_cluster": int(target_cluster),
            "proxy_cluster": int(proxy_cluster),
            "atk_epochs": int(atk_epochs), "lr": float(lr),
            "count": int(count), "count2": int(count2),
            "alpha": float(alpha), "attack_seed": int(attack_seed),
            "guided_centroid_lambda": 1.0,
        }, f, indent=2)
    attacks_dataloader1(dataloader, server, target_cluster, proxy_cluster,
                        DEVICE, atk_epochs, lr, save_dir, np.asarray(embeds_s2c), labels)
    return save_dir


# ======================================================================
# 6. EVALUATION
# ======================================================================
@torch.no_grad()
def evaluate_attack(tag, model_dir, atk_dir, atk_epoch,
                    victim_class, proxy_class, test_classes=None,
                    sb=SPLIT_BLOCK):
    """Measure ACC-V, NVA Drop, MP and per-class accuracy on the clean balanced
    test set using: clean net1 + clean net3 + ATTACKED server.

    If `test_classes` is given, restrict the evaluation to those true classes
    (used by the dynamic-distribution experiment to evaluate within a phase)."""
    net1, server, net3 = create_models(sb)
    net1.load_state_dict(torch.load(os.path.join(model_dir, "net1_nor_last.pt")))
    server.load_state_dict(torch.load(os.path.join(atk_dir, f"server_{atk_epoch}.pt")))
    net3.load_state_dict(torch.load(os.path.join(model_dir, "net3_nor_last.pt")))
    net1.eval(); server.eval(); net3.eval()

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=get_transforms(False))
    if test_classes is not None:
        tgt = np.array(testset.targets)
        keep = np.where(np.isin(tgt, np.array(test_classes)))[0]
        testset = Subset(testset, keep.tolist())
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


def nva_drop(per_class_pre, per_class_post, victim_class):
    """NVA Drop = mean non-victim accuracy (pre) - (post), averaged over the 9
    non-victim classes (matches the paper's benign-accuracy-drop definition)."""
    others = [c for c in range(NUM_CLASSES) if c != victim_class]
    pre = np.mean([per_class_pre[c] for c in others])
    post = np.mean([per_class_post[c] for c in others])
    return float(pre - post), float(pre), float(post)


# ======================================================================
# Convenience: pick a proxy class = nearest cluster's majority class != victim
# ======================================================================
def choose_proxy_class(cluster_to_class, victim_class):
    for c, cls in sorted(cluster_to_class.items()):
        if cls != victim_class:
            return c, cls
    return None, None
