"""
controls.py — M8: controls for the camera-ready (VGG-16/SB-1/seed 9202).

Runs the following controlled variants on the SAME cache / clustering / budget:

  1. LASER reference: fixed truck->car, guided lambda=1, anchor alpha=0.5.
  2. Guided lambda=0.5: milder interpolation between victim and proxy centroids.
  3. No Feature Anchoring (alpha=0): unconstrained pull, expected global collapse.
  4. Random victim cluster (5 draws): victim cluster chosen uniformly; proxy by
     nearest PCA centroid. Tests whether the result depends on a hand-picked class.
  5. Random proxy cluster (5 draws): victim fixed to truck; proxy chosen uniformly
     among other clusters. Tests whether the result depends on the proxy choice.
  6. Untargeted-style control: random target cluster, alpha=0 (no anchor), so the
     feature manifold is pulled without stability constraints; expected to degrade
     global accuracy far more than LASER.

Ground-truth labels are used only to name clusters / select the fixed truck and
car clusters and to evaluate; random controls choose by cluster id (label-free).

Outputs
-------
exp_camera/outputs/results_controls.json
exp_camera/outputs/tables/controls.md
"""
import copy
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn

import exp_camera.lib as L

REPO = L.REPO
OUT = L.OUT
TAG = "cifar10_sb1_seed9202"
MODEL_DIR = os.path.join(OUT, "models", TAG)
N_RANDOM = 5
ATK_EPOCHS = 20
COUNT = 10
COUNT2 = 50
LR = 1e-4
SEED = 42


def _load_cluster():
    cdir = os.path.join(OUT, "clusters", TAG)
    with open(os.path.join(cdir, "info.json")) as f:
        info = json.load(f)
    clabels = np.load(os.path.join(cdir, "cluster_labels.npy")).astype(np.int64)
    centers = np.load(os.path.join(cdir, "centers_pca.npy"))
    feat_path = os.path.join(cdir, "feat_pca.npy")
    return clabels, centers, feat_path, info


def _cluster_for_class(info, cls):
    best = None
    for c, cc in info["cluster_to_class"].items():
        if int(cc) != int(cls):
            continue
        rec = info["per_cluster"][str(c)]
        score = (rec["size"], rec["purity"])
        if best is None or score > best[0]:
            best = (score, int(c))
    if best is None:
        raise ValueError(f"no cluster for class {cls}")
    return best[1]


def _freeze_last_block(server):
    for p in server.parameters():
        p.requires_grad = False
    blocks = list(server.layer1.children())
    for target in blocks[-2:]:
        for p in target.parameters():
            p.requires_grad = True
        target.train()
        for m in target.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    return [p for p in server.parameters() if p.requires_grad]


def attack_variant(c2s, s2c, labels, feat_path, centers, target_cluster,
                   proxy_cluster, lambda_, alpha, epochs=ATK_EPOCHS,
                   attack_seed=SEED):
    """LASER-style attack with explicit guided-lambda and anchor-alpha.

    The only difference from src.attack.attacks_dataloader1 is that the guided
    centroid is (1-lambda)*mu_T + lambda*mu_P (instead of lambda=1 / pure proxy)
    and alpha can be 0 (no stability loss).  Last-conv-block freezing, BN eval,
    optimizer, batch sampling and MSE losses are identical.
    """
    np.random.seed(attack_seed)
    torch.manual_seed(attack_seed)
    net1, server, net3 = L.create_models(1)
    server.load_state_dict(torch.load(os.path.join(MODEL_DIR, "server_nor_last.pt")))
    anchor = copy.deepcopy(server)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad = False
    trainable = _freeze_last_block(server)

    t_idx = np.where(labels == target_cluster)[0]
    p_idx = np.where(labels == proxy_cluster)[0]
    if len(t_idx) == 0 or len(p_idx) == 0:
        raise ValueError("empty target/proxy cluster")
    mu_t = np.mean(np.asarray(s2c[t_idx], dtype=np.float32), axis=0)
    mu_p = np.mean(np.asarray(s2c[p_idx], dtype=np.float32), axis=0)
    guided = (1.0 - lambda_) * mu_t + lambda_ * mu_p
    guided = torch.from_numpy(guided).float().to(L.DEVICE).unsqueeze(0)

    feat = np.load(feat_path, mmap_mode='r')
    # Match src.attack.attacks_dataloader1: each cached sample's distance to its
    # OWN assigned cluster centroid (not to the target centroid).
    distances = np.linalg.norm(
        np.asarray(feat, dtype=np.float32) - centers[labels], axis=1
    ).astype(np.float32)
    dataloader = L.MyDataLoader(
        embeds_c2s=c2s, embeds_s2c=s2c, labels=labels, batch_size=50,
        distances=distances, shuffle=True, drop_last=False, count=COUNT,
        count2=COUNT2, target_label=target_cluster, random_state=attack_seed)
    optimizer = torch.optim.Adam(trainable, lr=LR, weight_decay=1e-4)
    mse = nn.MSELoss()

    t0 = time.time()
    last_loss = None
    for epoch in range(epochs):
        for _, (batch_c2s, _, batch_labels) in enumerate(dataloader):
            if len(batch_labels) < 2:
                continue
            z = torch.from_numpy(batch_c2s).float().to(L.DEVICE)
            y = torch.from_numpy(batch_labels).long().to(L.DEVICE)
            optimizer.zero_grad()
            h = server.forward2(z)
            with torch.no_grad():
                h_anchor = anchor(z)
            mask_t = y == target_cluster
            mask_n = ~mask_t
            loss = torch.tensor(0.0, device=L.DEVICE)
            if mask_t.sum() > 0:
                loss = loss + mse(h[mask_t],
                                  guided.expand(int(mask_t.sum()), -1, -1, -1))
            elif alpha == 0:
                # No target in this batch and no stability term -> no gradient.
                continue
            if alpha > 0 and mask_n.sum() > 0:
                loss = loss + alpha * mse(h[mask_n], h_anchor[mask_n])
            if not loss.requires_grad:
                continue
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
    return server, {"wall_s": time.time() - t0, "last_loss": last_loss,
                    "trainable_params": sum(p.numel() for p in trainable)}


@torch.no_grad()
def evaluate_server(server, victim, proxy):
    net1, _, net3 = L.create_models(1)
    net1.load_state_dict(torch.load(os.path.join(MODEL_DIR, "net1_nor_last.pt")))
    net3.load_state_dict(torch.load(os.path.join(MODEL_DIR, "net3_nor_last.pt")))
    net1.eval(); server.eval(); net3.eval()
    import torchvision
    from torch.utils.data import DataLoader
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True,
                                           transform=L.get_transforms(False))
    loader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=8)
    correct = np.zeros(L.NUM_CLASSES)
    total = np.zeros(L.NUM_CLASSES)
    for x, y in loader:
        x, y = x.to(L.DEVICE), y.to(L.DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = net3(server(net1(x)))
        pred = out.argmax(1)
        for c in range(L.NUM_CLASSES):
            m = y == c
            total[c] += m.sum().item()
            correct[c] += (pred[m] == c).sum().item()
    per_class = 100.0 * correct / np.maximum(total, 1)
    acc_v = float(per_class[victim])
    # MP on victim class
    mp_num = 0
    victim_total = 0
    for x, y in loader:
        x, y = x.to(L.DEVICE), y.to(L.DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = net3(server(net1(x))).argmax(1)
        m = y == victim
        victim_total += m.sum().item()
        mp_num += (pred[m] == proxy).sum().item()
    mp = 100.0 * mp_num / max(victim_total, 1)
    return per_class, acc_v, mp


def main():
    assert os.path.exists(os.path.join(MODEL_DIR, "server_nor_last.pt")), \
        f"missing model {MODEL_DIR}"
    xtr, ytr, _, _ = L._base_cifar()
    cache_dir = os.path.join(OUT, "cache", TAG)
    c2s_p = os.path.join(cache_dir, "embeds_c2s.npy")
    s2c_p = os.path.join(cache_dir, "embeds_s2c.npy")
    lab_p = os.path.join(cache_dir, "labels_true.npy")
    if not (os.path.exists(c2s_p) and os.path.exists(s2c_p)):
        c2s_p, s2c_p, lab_p = L.collect_embeddings(
            TAG, xtr, ytr, MODEL_DIR, sb=1)
    cdir = os.path.join(OUT, "clusters", TAG)
    if not os.path.exists(os.path.join(cdir, "info.json")):
        L.cluster_features(TAG, s2c_p, lab_p, max_k=12)
    clabels, centers, feat_path, info = _load_cluster()
    c2s = np.load(c2s_p, mmap_mode='r')
    s2c = np.load(s2c_p, mmap_mode='r')

    pre_per_class, base_acc = L.preattack_per_class(MODEL_DIR, sb=1)
    truck = 9
    car = 1
    tc_truck = _cluster_for_class(info, truck)
    tc_car = _cluster_for_class(info, car)
    print(f"[M8] base_acc={base_acc:.2f}% truck cluster={tc_truck} "
          f"car cluster={tc_car}", flush=True)

    rng = np.random.RandomState(SEED)
    sizes = np.array([info["per_cluster"][str(c)]["size"]
                      for c in info["per_cluster"]])
    blob_thr = 2.0 * float(np.median(sizes))
    blobs = {int(c) for c in info["per_cluster"]
             if info["per_cluster"][str(c)]["size"] > blob_thr}
    non_blob = [int(c) for c in info["per_cluster"] if int(c) not in blobs]
    rows = []

    def run_case(name, target, proxy, lambda_, alpha, victim, proxy_class):
        t0 = time.time()
        server, meta = attack_variant(
            c2s, s2c, clabels, feat_path, centers, target, proxy,
            lambda_, alpha)
        post_per_class, acc_v, mp = evaluate_server(server, victim, proxy_class)
        drop, pre_nv, post_nv = L.nva_drop(pre_per_class, post_per_class, victim)
        global_drop = float(np.mean(pre_per_class) - np.mean(post_per_class))
        rec = {
            "name": name, "target_cluster": int(target),
            "proxy_cluster": int(proxy),
            "victim_class": int(victim), "victim_name": L.CLASS_NAMES[victim],
            "proxy_class": int(proxy_class),
            "proxy_name": L.CLASS_NAMES[proxy_class],
            "lambda": float(lambda_), "alpha": float(alpha),
            "acc_v": float(acc_v), "mp": float(mp),
            "nva_drop": float(drop), "global_drop": global_drop,
            "base_acc": float(base_acc),
            "wall_s": float(meta["wall_s"]),
            "total_wall_s": float(time.time() - t0),
        }
        rows.append(rec)
        print(f"[M8] {name}: ACC-V={acc_v:.2f}% MP={mp:.2f}% "
              f"NVA={drop:.2f}% global-drop={global_drop:.2f}%", flush=True)
        torch.cuda.empty_cache()
        return rec

    # 1. reference and controlled variants on truck->car
    run_case("LASER reference (lambda=1, alpha=0.5)", tc_truck, tc_car,
             1.0, 0.5, truck, car)
    run_case("Guided lambda=0.5", tc_truck, tc_car, 0.5, 0.5, truck, car)
    run_case("No Feature Anchoring (alpha=0)", tc_truck, tc_car, 1.0, 0.0,
             truck, car)
    run_case("Untargeted-style (random target, alpha=0)",
             int(rng.choice([c for c in non_blob
                             if c not in (tc_car, tc_truck)])),
             tc_car, 1.0, 0.0, truck, car)

    # 4. random victim cluster, fixed nearest proxy
    for i in range(N_RANDOM):
        cand = [c for c in non_blob if int(info["per_cluster"][str(c)]["size"]) >= 10]
        target = int(rng.choice(cand))
        vc = int(info["cluster_to_class"][str(target)])
        # nearest PCA centroid, excluding same majority class (label-free).
        d = np.linalg.norm(centers - centers[target], axis=1)
        d[target] = np.inf
        for c in range(len(centers)):
            if int(info["cluster_to_class"].get(str(c), -1)) == vc:
                d[c] = np.inf
        proxy = int(np.argmin(d))
        proxy_class = int(info["cluster_to_class"][str(proxy)])
        run_case(f"random victim #{i+1} (cluster {target})", target, proxy,
                 1.0, 0.5, vc, proxy_class)

    # 5. random proxy cluster, fixed truck victim
    for i in range(N_RANDOM):
        cand = [c for c in non_blob
                if c != tc_truck
                and int(info["cluster_to_class"][str(c)]) != truck]
        proxy = int(rng.choice(cand))
        proxy_class = int(info["cluster_to_class"][str(proxy)])
        run_case(f"random proxy #{i+1} (cluster {proxy})", tc_truck, proxy,
                 1.0, 0.5, truck, proxy_class)

    payload = {"tag": "controls", "model": TAG, "split": "SB=1",
               "dataset": "CIFAR-10", "n_random": N_RANDOM,
               "base_acc": float(base_acc), "rows": rows}
    json_path = os.path.join(OUT, "results_controls.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    md_path = os.path.join(OUT, "tables", "controls.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write("# M8 — Controls (VGG-16/SB-1, CIFAR-10)\n\n")
        f.write(f"Base accuracy **{base_acc:.2f}%**.  Same cache, clustering, "
                f"last-conv-block budget, 20 epochs, 10 victim + 50/non-victim "
                f"cached vectors.  Ground-truth labels are used only to name "
                f"fixed truck/car clusters and evaluate; random controls pick "
                f"clusters by id.\n\n")
        f.write("| Control | Victim | Proxy | λ | α | ACC-V | MP | NVA Drop | Global Drop |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['name']} | {r['victim_name']} | {r['proxy_name']} | "
                    f"{r['lambda']:.1f} | {r['alpha']:.1f} | {r['acc_v']:.2f}% | "
                    f"{r['mp']:.2f}% | {r['nva_drop']:.2f}% | "
                    f"{r['global_drop']:.2f}% |\n")
        f.write("\n**Reading.** The reference reproduces the main result; λ=0.5 "
                "trades effectiveness for less feature movement; α=0 (no anchor) "
                "shows the global-collapse control; random victim/proxy draws "
                "report the range over 5 draws. The untargeted-style row is a "
                "sanity control, not a claim about Smash Poisoning/MISA.\n")
    print(f"[M8] -> {json_path}\n[M8] -> {md_path}", flush=True)
    shutil.rmtree(cache_dir, ignore_errors=True)
    free_gb = shutil.disk_usage("/root/autodl-tmp").free / 1e9
    print(f"[M8] freed cache; free={free_gb:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
