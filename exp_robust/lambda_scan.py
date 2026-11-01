"""
lambda_scan.py — λ (steering-coefficient) sensitivity scan for LASER on
VGG16 / SB=1 / CIFAR-10, victim truck(9) → proxy car(1).

Answers RAID 2026 reviewer 296B's "across architectures" ask for the centroid-
interpolation mechanism: the paper's Appendix B λ table is ResNeXt-50 only;
this produces the directly comparable VGG-16 table.

KEY POINT: the attack in actual use, src/attack.py::attacks_dataloader1, has NO
λ knob — it pulls victim features straight to the proxy centroid μ_P, i.e. the
λ=1 (saturated) case. The paper's Eq.(3) guided centroid
    μ_guided(λ) = (1−λ)·μ_T + λ·μ_P
is not implemented there. So this module adds μ_T (victim-cluster centroid) and
the interpolation, then sweeps λ. It does NOT modify src/attack.py (the paper's
original code stays intact). At λ=1.0 this reproduces the existing baseline
attack → built-in sanity check (truck→car: ACC-V≈0, MP≈99, NVA≈0.6).

Reuses (no retraining): models/baseline, clusters/baseline. Regenerates the
~10 GB embedding cache ONCE (shared by all λ), then deletes it.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import copy
import json
import shutil
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

import exp_robust.lib as L
from exp_robust.lib import OUT, NUM_CLASSES, CLASS_NAMES
from exp_robust.experiments import preattack_per_class
from src.MyDataLoader import MyDataLoader

# ---------------------------------------------------------------- config
LAMBDAS = [0.50, 0.55, 0.57, 0.59, 0.80, 1.00]   # exact match to paper Appendix B
VICTIM_CLASS = 9        # truck  (true label)
PROXY_CLASS = 1         # car    (true label)
TARGET_CLUSTER = 3      # cluster whose majority true-class == truck
PROXY_CLUSTER = 1       # cluster whose majority true-class == car
ALPHA = 0.5
ATK_EPOCHS = 20
COUNT = 10              # victim cached samples
COUNT2 = 50             # per-non-victim cached samples
LR = 1e-4
SEED = 12345            # fixed RNG for the centroid draws -> reproducible scan


def regime_of_acc(acc_v):
    """Data-driven regime label (by observed Victim Accuracy, not by a fixed λ
    threshold). The paper's Appendix B thresholds are ResNeXt-specific; VGG16
    saturates at a lower λ, so the regime must be read off the behavior."""
    if acc_v > 10.0:
        return "I (Insufficient)"
    if acc_v > 1.0:
        return "II (Transitional)"
    return "III (Saturated)"


# ---------------------------------------------------------------- λ-aware attack
def laser_attack_lambda(dataloader, server, target_label, close_label,
                        device, attack_epochs, learning_rate, save_dir,
                        guided_centroid):
    """LASER attack with a *precomputed* guided centroid.

    Identical to src/attack.py::attacks_dataloader1 except the guided centroid
    is passed in (so the caller can form (1−λ)·μ_T + λ·μ_P). VGG last-conv-
    block unfreeze, BN frozen, anchor MSE, α=0.5, Adam — all unchanged, so
    λ=1 reproduces the paper attack exactly.
    """
    os.makedirs(save_dir, exist_ok=True)
    alpha = ALPHA

    server.reset()
    anchor_server = copy.deepcopy(server)
    anchor_server.eval()
    for p in anchor_server.parameters():
        p.requires_grad = False

    # freeze all, then unfreeze the last 2 server blocks (last conv block + pool)
    for param in server.parameters():
        param.requires_grad = False
    server_blocks = list(server.layer1.children())
    unfreeze_count = 2
    print(f"[lambda-scan] NetServer has {len(server_blocks)} blocks; "
          f"unfreezing last {unfreeze_count}.", flush=True)
    for i in range(1, unfreeze_count + 1):
        target_block = server_blocks[-i]
        for param in target_block.parameters():
            param.requires_grad = True
        target_block.train()
        for m in target_block.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = [p for p in server.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable params were unfrozen.")
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)
    loss_mse = nn.MSELoss()

    for epoch in trange(attack_epochs):
        for _, (batch_c2s, _, batch_labels) in enumerate(dataloader):
            if len(batch_labels) < 2:
                continue
            batch_c2s = torch.from_numpy(batch_c2s).float().to(device)
            batch_labels = torch.from_numpy(batch_labels).long().to(device)
            batch_c2s.requires_grad_(False)

            optimizer.zero_grad()
            embed_current = server.forward2(batch_c2s)
            with torch.no_grad():
                embed_anchor = anchor_server(batch_c2s)

            target_mask = (batch_labels == target_label)
            non_target_mask = ~target_mask
            loss = 0

            if target_mask.sum() > 0:
                target_features = embed_current[target_mask]
                cur = guided_centroid.expand(target_features.size(0), -1, -1, -1)
                l_atk = loss_mse(target_features, cur)
                loss += l_atk
            if non_target_mask.sum() > 0:
                stable_features = embed_current[non_target_mask]
                anchor_features = embed_anchor[non_target_mask]
                l_stable = loss_mse(stable_features, anchor_features)
                loss += alpha * l_stable

            loss.backward()
            optimizer.step()

        torch.save(server.state_dict(), os.path.join(save_dir, f"server_{epoch+1}.pt"))

    return save_dir


def compute_centroid(embeds_s2c, labels, label, max_n=1000, rng=None):
    """Mean of up to `max_n` server-output features whose (cluster) label ==
    `label`. Deterministic when `rng` is fixed."""
    idx = np.where(labels == label)[0]
    if len(idx) == 0:
        raise ValueError(f"No samples with label {label}")
    if rng is not None:
        idx = rng.choice(idx, min(len(idx), max_n), replace=False)
    elif len(idx) > max_n:
        idx = idx[:max_n]
    return np.mean(embeds_s2c[idx], axis=0)


def main():
    model_dir = os.path.join(OUT, "models", "baseline")
    cache_dir = os.path.join(OUT, "cache", "baseline")
    c2s_p = os.path.join(cache_dir, "embeds_c2s.npy")
    s2c_p = os.path.join(cache_dir, "embeds_s2c.npy")
    lab_p = os.path.join(cache_dir, "labels_true.npy")

    # ---- 1. cache embeddings (regenerate only if missing) -----------------
    if not (os.path.exists(c2s_p) and os.path.exists(s2c_p) and os.path.exists(lab_p)):
        print("[lambda-scan] regenerating baseline embedding cache (~10 GB) ...",
              flush=True)
        xtr, ytr, _, _ = L._base_cifar()           # clean CIFAR-10 train
        c2s_p, s2c_p, lab_p = L.collect_embeddings("baseline", xtr, ytr, model_dir)
    else:
        print("[lambda-scan] reusing existing baseline cache.", flush=True)

    # ---- 2. load existing clustering (clusters/baseline, k*=10) ----------
    cluster_dir = os.path.join(OUT, "clusters", "baseline")
    clabels = np.load(os.path.join(cluster_dir, "cluster_labels.npy")).astype(np.int64)
    feat_path = os.path.join(cluster_dir, "feat_pca.npy")
    centers = np.load(os.path.join(cluster_dir, "centers_pca.npy"))
    with open(os.path.join(cluster_dir, "info.json")) as f:
        info = json.load(f)
    print(f"[lambda-scan] clustering: k*={info['k_star']} ARI={info['ari']:.3f} "
          f"purity={info['purity']:.3f}", flush=True)

    # ---- 3. pre-attack per-class accuracy (for NVA-Drop) ------------------
    pre_per_class, base_acc = preattack_per_class(model_dir)
    print(f"[lambda-scan] base acc = {base_acc:.2f}%  "
          f"victim({CLASS_NAMES[VICTIM_CLASS]}) pre = {pre_per_class[VICTIM_CLASS]:.2f}%",
          flush=True)

    # ---- 4. compute μ_T and μ_P ONCE (deterministic) ---------------------
    embeds_s2c = np.load(s2c_p, mmap_mode='r')
    embeds_c2s = np.load(c2s_p, mmap_mode='r')
    labels = clabels
    rng = np.random.RandomState(SEED)
    mu_T = compute_centroid(embeds_s2c, labels, TARGET_CLUSTER, rng=rng)  # victim
    mu_P = compute_centroid(embeds_s2c, labels, PROXY_CLUSTER, rng=rng)   # proxy
    feat = np.load(feat_path, mmap_mode='r')
    distances = np.linalg.norm(
        np.asarray(feat, dtype=np.float32) - centers[labels], axis=1
    ).astype(np.float32)

    # ---- 5. sweep λ -------------------------------------------------------
    rows = []
    for lam in LAMBDAS:
        print(f"\n{'='*60}\n[lambda-scan] λ = {lam}\n{'='*60}",
              flush=True)
        guided = (1.0 - lam) * mu_T + lam * mu_P
        guided_t = torch.from_numpy(guided).float().to(L.DEVICE).unsqueeze(0)

        net1, server, net3 = L.create_models()
        server.load_state_dict(torch.load(
            os.path.join(model_dir, "server_nor_last.pt")))

        dataloader = MyDataLoader(
            embeds_c2s=embeds_c2s, embeds_s2c=embeds_s2c, labels=labels,
            batch_size=50, distances=distances, shuffle=True, drop_last=False,
            count=COUNT, count2=COUNT2, target_label=TARGET_CLUSTER)

        save_dir = os.path.join(OUT, "attacks", f"lambda_scan_l{lam:.2f}")
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        laser_attack_lambda(dataloader, server, TARGET_CLUSTER, PROXY_CLUSTER,
                            L.DEVICE, ATK_EPOCHS, LR, save_dir, guided_t)

        post_per_class, acc_v, mp = L.evaluate_attack(
            "baseline", model_dir, save_dir, ATK_EPOCHS, VICTIM_CLASS, PROXY_CLASS)
        drop, pre_nv, post_nv = L.nva_drop(pre_per_class, post_per_class, VICTIM_CLASS)

        rows.append({
            "lambda": lam,
            "regime": regime_of_acc(acc_v),
            "acc_v": float(acc_v),
            "nva_drop": float(drop),
            "mp": float(mp),
            "nonvictim_acc_pre": float(pre_nv),
            "nonvictim_acc_post": float(post_nv),
        })
        print(f"[lambda-scan] λ={lam:.2f} -> ACC-V={acc_v:.2f}%  "
              f"NVA-Drop={drop:.2f}%  MP={mp:.2f}%", flush=True)

        # incremental save (crash-safe) + free this λ's artifacts
        _save(rows, base_acc, info)
        shutil.rmtree(save_dir, ignore_errors=True)
        del net1, server, net3
        torch.cuda.empty_cache()

    # ---- 6. sanity check vs baseline (λ=1.0) -----------------------------
    sanity = None
    baseline_json = os.path.join(OUT, "results_baseline.json")
    if os.path.exists(baseline_json):
        with open(baseline_json) as f:
            base = json.load(f)
        for a in base.get("attacks", []):
            if a.get("victim_class") == VICTIM_CLASS:
                row1 = next(r for r in rows if r["lambda"] == 1.0)
                sanity = {
                    "baseline_lambda1": {k: a[k] for k in ("acc_v", "mp", "nva_drop")},
                    "scan_lambda1": {k: row1[k] for k in ("acc_v", "mp", "nva_drop")},
                    "note": "λ=1.0 is the saturated regime; the scan reproduces "
                            "the baseline attack (μ_P only) up to the random "
                            "centroid draw and training stochasticity.",
                }
                break
    _save(rows, base_acc, info, sanity=sanity)

    # ---- 7. free the 10 GB cache -----------------------------------------
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[lambda-scan] freed cache {cache_dir}", flush=True)

    _write_table(rows, base_acc)
    print("\n[lambda-scan] DONE.", flush=True)


def _save(rows, base_acc, info, sanity=None):
    out = {
        "tag": "lambda_scan",
        "config": {
            "model": "VGG16", "split": "SB=1", "dataset": "CIFAR-10",
            "victim_class": VICTIM_CLASS, "victim_name": CLASS_NAMES[VICTIM_CLASS],
            "proxy_class": PROXY_CLASS, "proxy_name": CLASS_NAMES[PROXY_CLASS],
            "target_cluster": TARGET_CLUSTER, "proxy_cluster": PROXY_CLUSTER,
            "alpha": ALPHA, "attack_epochs": ATK_EPOCHS,
            "count": COUNT, "count2": COUNT2, "lr": LR,
        },
        "base_acc": float(base_acc),
        "clustering": {k: info[k] for k in ("k_star", "ari", "nmi", "purity")},
        "rows": rows,
    }
    if sanity is not None:
        out["sanity_check"] = sanity
    path = os.path.join(OUT, "results_lambda_scan.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def _write_table(rows, base_acc):
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    path = os.path.join(OUT, "tables", "lambda_scan.md")
    with open(path, "w") as f:
        f.write("# λ Scan — VGG16, SB=1, CIFAR-10, victim truck(9) → proxy car(1)\n\n")
        f.write(f"Base accuracy **{base_acc:.2f}%**. α=0.5, 20 epochs, "
                "10 victim / 50-per-non-victim cached samples. "
                "Guided centroid μ=(1−λ)μ_T+λμ_P.\n\n")
        f.write("| Regime | λ | ACC-V | NVA-Drop | MP |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['regime']} | {r['lambda']:.2f} | "
                    f"{r['acc_v']:.2f}% | {r['nva_drop']:.2f}% | {r['mp']:.2f}% |\n")
    print(f"[lambda-scan] table -> {path}", flush=True)


if __name__ == "__main__":
    main()
