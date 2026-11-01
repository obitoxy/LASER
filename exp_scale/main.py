"""
main.py - Orchestrator for the LASER scale experiment (CIFAR-100, 100 classes).

Answers RAID 2026 reviewers 296A #4 / 296B #5 ("larger tasks"): does the
label-free LASER pipeline still (a) recover the right cluster count and
(b) surgically destroy a victim cluster when K grows from 10 to 100?

Usage (under conda env `cdssc`):
    python -m exp_scale.main train cifar100          # ~25 min: train VGG16/SB-1
    python -m exp_scale.main run cifar100            # cache + cluster + 12 blind attacks
    python -m exp_scale.main run cifar100 20         # ... attack 20 blind clusters

Design (control-variable): vs the exp_robust CIFAR-10 baseline, exactly ONE
thing changes - the label-space size K (10 -> 100). Architecture (VGG16), split
(SB=1), resolution (32x32), K-search method, LASER mechanism, and the cached
attack budget (10 victim + 50/non-victim) are all held fixed. The only code
change is the K-search ceiling max_k 12 -> 120 (see lib.py).

Blind cluster selection: the attacker picks victims BY CLUSTER ID using a
label-free rule (evenly-spaced over the size-ranked clusters, so the sample
spans small->large) and the proxy by nearest PCA centroid. Ground-truth labels
are an EVALUATION ORACLE ONLY - they name each cluster's majority class and
measure recall collapse; they never select or execute the attack. We attack a
representative subset (default 12 of 100) rather than all 100 purely for
wall-clock; the per-cluster outcomes + aggregate stats are reported.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import sys
import json
import shutil
import numpy as np
import torch

import exp_scale.lib as L
from exp_scale.lib import OUT, NUM_CLASSES, CLASS_NAMES, MAX_K

TAG = "cifar100"              # namespace for cache / clusters / attacks / models
MODEL_TAG = "cifar100"       # the trained model lives under models/<MODEL_TAG>
EPOCHS = 120                 # match the exp_robust baseline protocol
ATK_EPOCHS = 20              # match the paper's reference attack
COUNT = 10                   # victim cached samples (paper setting)
COUNT2 = 50                  # per-non-victim cached samples (paper setting)
LR = 1e-4
DEFAULT_N_BLIND = 12         # default #clusters to attack blindly
DESTROY_RECALL_THRESH = 1.0  # recall <= this (%) counts as "destroyed"


# ----------------------------------------------------------------------
def train(tag=TAG, epochs=EPOCHS):
    xtr, ytr, _, _ = L._base_cifar100()
    print(f"=== TRAIN {tag}: clean balanced CIFAR-100 (50000, 100 classes) ===",
          flush=True)
    L.train_split_model(tag, xtr, ytr, epochs=epochs)


def _select_blind_clusters(info, n, exclude=None):
    """Label-free victim selection: pick `n` clusters evenly spaced over the
    size-ranked cluster list (spans small->large). Size is observable without
    labels; the majority-class identity of each selected cluster is looked up
    only AFTER selection, as the evaluation oracle. `exclude` (a set of cluster
    ids, e.g. the garbage blobs) is skipped - those are not separable semantic
    clusters and are excluded label-free, consistent with the §6.3 threat model."""
    exclude = exclude or set()
    pcs = info["per_cluster"]
    sizes = sorted(((int(c), pcs[c]["size"]) for c in pcs if int(c) not in exclude),
                   key=lambda x: x[1])
    if len(sizes) <= n:
        sel = [c for c, _ in sizes]
    else:
        idx = np.linspace(0, len(sizes) - 1, n).round().astype(int)
        idx = sorted(set(idx.tolist()))          # dedup at small n
        sel = [sizes[i][0] for i in idx]
    print(f"[{TAG}] blind-selected {len(sel)} clusters by size-rank stratification "
          f"(label-free); sizes={[dict(sizes)[c] for c in sel]}", flush=True)
    return sel


def run(tag=TAG, n_blind=DEFAULT_N_BLIND, atk_epochs=ATK_EPOCHS):
    model_dir = os.path.join(OUT, "models", MODEL_TAG)
    assert os.path.exists(os.path.join(model_dir, "server_nor_last.pt")), \
        f"model not trained for {MODEL_TAG}; run 'train {MODEL_TAG}' first"

    cache_dir = os.path.join(OUT, "cache", tag)
    c2s_p = os.path.join(cache_dir, "embeds_c2s.npy")
    s2c_p = os.path.join(cache_dir, "embeds_s2c.npy")
    lab_p = os.path.join(cache_dir, "labels_true.npy")

    # ---- 1. cache embeddings (reuse if present) ---------------------------
    if not (os.path.exists(c2s_p) and os.path.exists(s2c_p) and os.path.exists(lab_p)):
        print(f"[{TAG}] building embedding cache (~10 GB) under cache/{tag} ...",
              flush=True)
        xtr, ytr, _, _ = L._base_cifar100()
        c2s_p, s2c_p, lab_p = L.collect_embeddings(tag, xtr, ytr, model_dir)
    else:
        print(f"[{TAG}] reusing existing cache/{tag}.", flush=True)

    # ---- 2. cluster (reuse if present; max_k=120) -------------------------
    cluster_dir = os.path.join(OUT, "clusters", tag)
    info_path = os.path.join(cluster_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        clabels = np.load(os.path.join(cluster_dir, "cluster_labels.npy")).astype(np.int64)
        centers = np.load(os.path.join(cluster_dir, "centers_pca.npy"))
        feat_path = os.path.join(cluster_dir, "feat_pca.npy")
        print(f"[{TAG}] reusing clustering: k*={info['k_star']} "
              f"(max_k={info.get('max_k', MAX_K)}) ARI={info['ari']:.3f} "
              f"NMI={info['nmi']:.3f} purity={info['purity']:.3f}", flush=True)
    else:
        clabels, centers, feat_path, info = L.cluster_features(
            tag, s2c_p, lab_p, max_k=MAX_K)
        with open(info_path) as f:
            info = json.load(f)   # reload (JSON-serialized keys)

    c2c = info["cluster_to_class"]               # str(cluster) -> int(class)

    # ---- 3. pre-attack per-class accuracy (oracle, for NVA-Drop + recall) -
    pre_per_class, base_acc = L.preattack_per_class(model_dir)
    print(f"[{TAG}] base acc = {base_acc:.2f}%", flush=True)

    # ---- 4. blind-select + attack each chosen cluster ---------------------
    # Label-free guard: exclude over-clustering garbage blobs (anomalously large
    # central clusters, not separable semantic clusters) from BOTH victim and
    # proxy selection. Detectable without labels (size outlier). This is the
    # natural at-scale refinement of the nearest-centroid proxy: a blind attacker
    # does not redirect victims onto the giant central blended blob.
    blobs = L.blob_cluster_ids(info)
    if blobs:
        blob_names = [(b, CLASS_NAMES[int(c2c[str(b)])]) for b in blobs]
        blob_sizes = [info["per_cluster"][str(b)]["size"] for b in blobs]
        print(f"[{TAG}] label-free blob guard: excluding {len(blobs)} garbage "
              f"cluster(s) {list(zip(blob_names, blob_sizes))} (size > 2x median) "
              f"from victim/proxy selection.", flush=True)
    selected = _select_blind_clusters(info, n_blind, exclude=blobs)
    rows = []
    for tc in selected:
        vc = int(c2c[str(tc)])                    # victim majority class (oracle)
        vp = float(info["per_cluster"][str(tc)]["purity"])
        vsize = int(info["per_cluster"][str(tc)]["size"])
        pc = L.nearest_other_cluster(centers, tc, forbidden_class=vc,
                                     cluster_to_class=c2c,
                                     forbidden_clusters=blobs)
        pc_class = int(c2c[str(pc)])
        print(f"\n{'='*60}\n[{TAG}] BLIND cluster {tc} (maj class "
              f"{CLASS_NAMES[vc]}, purity {vp:.2f}, size {vsize}) -> "
              f"nearest-proxy cluster {pc} (maj class {CLASS_NAMES[pc_class]})\n"
              f"{'='*60}", flush=True)

        save_dir = L.run_laser_attack(
            tag, c2s_p, s2c_p, feat_path, centers, clabels,
            target_cluster=tc, proxy_cluster=pc,
            atk_epochs=atk_epochs, count=COUNT, count2=COUNT2, lr=LR,
            model_tag=MODEL_TAG)

        post_per_class, acc_v, mp = L.evaluate_attack(
            tag, model_dir, save_dir, atk_epochs,
            victim_class=vc, proxy_class=pc_class)
        drop, pre_nv, post_nv = L.nva_drop(pre_per_class, post_per_class, vc)

        rec = {
            "target_cluster": int(tc),
            "victim_class": vc, "victim_name": CLASS_NAMES[vc],
            "proxy_cluster": int(pc),
            "proxy_class": pc_class, "proxy_name": CLASS_NAMES[pc_class],
            "cluster_purity": vp, "cluster_size": vsize,
            "victim_recall_pre": float(pre_per_class[vc]),
            "victim_recall_post": float(acc_v),
            "mp": float(mp),
            "nva_drop": float(drop),
            "nonvictim_acc_pre": float(pre_nv),
            "nonvictim_acc_post": float(post_nv),
        }
        rows.append(rec)
        print(f"[{TAG}] cluster {tc} ({CLASS_NAMES[vc]}): "
              f"recall {pre_per_class[vc]:.1f}% -> {acc_v:.2f}%  "
              f"MP={mp:.2f}%  NVA-Drop={drop:.2f}%", flush=True)

        # crash-safe incremental save + free this cluster's ~1.2 GB checkpoints
        _save(rows, base_acc, info, atk_epochs=atk_epochs)
        shutil.rmtree(save_dir, ignore_errors=True)
        torch.cuda.empty_cache()

    # ---- 5. summary + table ----------------------------------------------
    n_destroyed = sum(1 for r in rows if r["victim_recall_post"] <= DESTROY_RECALL_THRESH)
    summary = {
        "dataset": "CIFAR-100", "num_classes": NUM_CLASSES, "max_k": MAX_K,
        "k_star": int(info["k_star"]),
        "clustering": {k: info[k] for k in ("ari", "nmi", "purity", "max_k")},
        "base_acc": float(base_acc),
        "clusters_attacked": len(rows),
        "clusters_destroyed_recall_le_1pct": n_destroyed,
        "n_garbage_blobs_excluded": len(blobs),
        "mean_nva_drop": float(np.mean([r["nva_drop"] for r in rows])) if rows else 0.0,
        "max_nva_drop": float(np.max([r["nva_drop"] for r in rows])) if rows else 0.0,
        "mean_mp": float(np.mean([r["mp"] for r in rows])) if rows else 0.0,
    }
    _save(rows, base_acc, info, summary=summary, atk_epochs=atk_epochs)
    _write_table(rows, base_acc, summary, info)

    # ---- 6. free the ~10 GB cache ----------------------------------------
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[{TAG}] freed cache {cache_dir}", flush=True)

    print(f"\n[{TAG}] DONE: clustering k*={info['k_star']} (ARI={info['ari']:.3f}, "
          f"purity={info['purity']:.3f}); {n_destroyed}/{len(rows)} blind clusters "
          f"destroyed (recall <= {DESTROY_RECALL_THRESH}%), "
          f"mean NVA-Drop={summary['mean_nva_drop']:.2f}%, "
          f"mean MP={summary['mean_mp']:.2f}%.", flush=True)


def _save(rows, base_acc, info, summary=None, atk_epochs=ATK_EPOCHS):
    out = {
        "tag": "scale_cifar100",
        "config": {
            "model": "VGG16", "split": "SB=1", "dataset": "CIFAR-100",
            "num_classes": NUM_CLASSES, "max_k": MAX_K,
            "victim_selection": "BLIND (cluster id, size-rank stratified; label-free; "
                               "garbage blobs excluded)",
            "proxy_selection": "nearest PCA centroid (label-free; garbage blobs excluded)",
            "blob_guard": "label-free: clusters with size > 2x median excluded from "
                          "victim/proxy selection (over-clustering central blobs are not "
                          "separable semantic clusters, per the §6.3 threat model)",
            "attack": "src.attack.attacks_dataloader1 (last-conv-block fine-tune, "
                      "guided-centroid pull + anchor, alpha=0.5)",
            "attack_epochs": atk_epochs, "count": COUNT, "count2": COUNT2, "lr": LR,
            "labels_role": "EVALUATION ORACLE ONLY (cluster naming + recall); "
                           "never used to select or execute the attack",
            "control_variable": "vs exp_robust CIFAR-10 baseline only K changes "
                                "(10->100); arch/split/resolution/method/budget fixed",
        },
        "base_acc": float(base_acc),
        "clustering": {k: info[k] for k in
                       ("k_star", "max_k", "ari", "nmi", "purity")},
        "rows": rows,
    }
    if summary is not None:
        out["summary"] = summary
    path = os.path.join(OUT, "results_scale_cifar100.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def _write_table(rows, base_acc, summary, info):
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    path = os.path.join(OUT, "tables", "scale_cifar100.md")
    with open(path, "w") as f:
        f.write("# Scale: CIFAR-100 (100 classes) - VGG16/SB-1\n\n")
        f.write("Blind (label-free) cluster selection: victim by cluster id "
                "(size-rank stratified), proxy by nearest centroid. Ground-truth "
                "labels name each cluster's majority class and measure recall "
                "collapse only. K-search ceiling raised to max_k=120 "
                "(was 12 for CIFAR-10). Base accuracy **%.2f%%**.\n\n" % base_acc)
        f.write(f"**Clustering:** K\\*={summary['k_star']} (true=100), "
                f"ARI={info['ari']:.3f}, NMI={info['nmi']:.3f}, "
                f"purity={info['purity']:.3f}.\n\n")
        f.write("| Cluster | Victim class (majority) | Purity | Size | "
                "Recall pre -> post | MP | NVA-Drop |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| %d | %s | %.2f | %d | %.1f -> **%.2f** | %.2f%% | %.2f%% |\n" % (
                r["target_cluster"], r["victim_name"], r["cluster_purity"],
                r["cluster_size"], r["victim_recall_pre"], r["victim_recall_post"],
                r["mp"], r["nva_drop"]))
        f.write("\n**Summary:** %d/%d blind clusters destroyed (recall <= 1%%), "
                "mean NVA-Drop = %.2f%%, max NVA-Drop = %.2f%%, mean MP = %.2f%%.\n" % (
                    summary["clusters_destroyed_recall_le_1pct"],
                    summary["clusters_attacked"],
                    summary["mean_nva_drop"], summary["max_nva_drop"],
                    summary["mean_mp"]))
    print(f"[{TAG}] table -> {path}", flush=True)


# ----------------------------------------------------------------------
def main():
    assert len(sys.argv) >= 3, \
        "usage: python -m exp_scale.main <train|run> <cifar100> [n_blind] [atk_epochs]"
    action, tag = sys.argv[1], sys.argv[2]
    if action == "train":
        train(tag)
    elif action == "run":
        n_blind = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_N_BLIND
        atk_epochs = int(sys.argv[4]) if len(sys.argv) > 4 else ATK_EPOCHS
        run(tag, n_blind, atk_epochs)
    else:
        raise SystemExit(f"unknown action {action}")


if __name__ == "__main__":
    main()
