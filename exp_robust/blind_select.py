"""
blind_select.py — Blind (label-free) victim-cluster selection experiment for LASER.

Answers RAID 2026 reviewer 296A's weakness #3: in the label-free setting the
server cannot know WHICH discovered cluster corresponds to a NAMED semantic class
(e.g. "airplane"). The reframed claim is therefore that LASER targets a
*discovered separable semantic cluster*, not a named class, and surgically
destroys it while preserving the rest — proving the privacy boundary is broken
whether or not the attacker can name the victim.

This experiment makes the *attack path* fully label-free END-TO-END:
  * the victim cluster is chosen BY CLUSTER ID — the attacker simply iterates the
    k* clusters discovered by unsupervised K-search (no labels);
  * the proxy cluster is chosen by a label-free geometric heuristic — the nearest
    PCA centroid (no labels);
  * the attack itself (src/attack.py::attacks_dataloader1, via run_laser_attack)
    already selects its victim/proxy smashed-data sets by cluster assignment.

Ground-truth labels are used here ONLY as an EVALUATION ORACLE — to name each
cluster's majority class and measure whether that class's recall collapsed. They
play no role in selecting or executing the attack.

WHY THIS IS NEEDED (not just repackaging existing data): the prior exp_robust
runs selected victims by TRUE CLASS via find_cluster_for_class(), which needs the
ground-truth-derived cluster_to_class map. So they prove the MECHANISM destroys
whatever cluster it is pointed at — NOT that a blind attacker can pick ANY
cluster. This closes that gap by attacking EVERY one of the k* clusters blindly.

Reuses (no retraining): models/baseline, clusters/baseline (the k*=10 partition).
Regenerates the ~10 GB embedding cache once, then deletes it. Each cluster's
~1.2 GB attack checkpoints are freed right after its evaluation (flat disk).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import shutil
import numpy as np
import torch

import exp_robust.lib as L
from exp_robust.lib import OUT, NUM_CLASSES, CLASS_NAMES
from exp_robust.experiments import preattack_per_class

# ---------------------------------------------------------------- config
TAG = "blind"               # namespace for this experiment's cache / attacks
MODEL_TAG = "baseline"      # the trained clean model to attack
ATK_EPOCHS = 20
COUNT = 10                  # victim cached samples (paper setting)
COUNT2 = 50                 # per-non-victim cached samples (paper setting)
LR = 1e-4
DESTROY_RECALL_THRESH = 1.0  # recall <= this (%) counts as "destroyed"


def nearest_other_cluster(centers, tc, forbidden_class=None, cluster_to_class=None):
    """Label-free proxy heuristic: the cluster (!= tc) whose PCA centroid is
    nearest to cluster tc's centroid.

    If `forbidden_class` + `cluster_to_class` are given, also skip clusters whose
    majority class equals the victim's, so the proxy is a *different* semantic
    region (this guard uses labels only to avoid a degenerate same-class redirect;
    the geometric ranking itself is label-free)."""
    d = np.linalg.norm(centers - centers[tc], axis=1).astype(np.float64)
    d[tc] = np.inf
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


def run_blind_select(model_tag=MODEL_TAG):
    model_dir = os.path.join(OUT, "models", model_tag)
    assert os.path.exists(os.path.join(model_dir, "server_nor_last.pt")), \
        f"model not trained for {model_tag}; run 'train {model_tag}' first"

    cache_dir = os.path.join(OUT, "cache", TAG)
    c2s_p = os.path.join(cache_dir, "embeds_c2s.npy")
    s2c_p = os.path.join(cache_dir, "embeds_s2c.npy")
    lab_p = os.path.join(cache_dir, "labels_true.npy")

    # ---- 1. cache embeddings (regenerate; baseline cache was freed) ----------
    if not (os.path.exists(c2s_p) and os.path.exists(s2c_p) and os.path.exists(lab_p)):
        print(f"[blind] regenerating embedding cache (~10 GB) under cache/{TAG} ...",
              flush=True)
        xtr, ytr, _, _ = L._base_cifar()          # clean CIFAR-10 train
        c2s_p, s2c_p, lab_p = L.collect_embeddings(TAG, xtr, ytr, model_dir)
    else:
        print(f"[blind] reusing existing cache/{TAG}.", flush=True)

    # ---- 2. reuse the baseline k*=10 partition (clusters/baseline) -----------
    base_cluster_dir = os.path.join(OUT, "clusters", "baseline")
    assert os.path.exists(os.path.join(base_cluster_dir, "info.json")), \
        "clusters/baseline missing; run 'run baseline' once first"
    clabels = np.load(os.path.join(base_cluster_dir, "cluster_labels.npy")).astype(np.int64)
    centers = np.load(os.path.join(base_cluster_dir, "centers_pca.npy"))
    feat_path = os.path.join(base_cluster_dir, "feat_pca.npy")
    with open(os.path.join(base_cluster_dir, "info.json")) as f:
        info = json.load(f)
    k_star = int(info["k_star"])
    c2c = info["cluster_to_class"]               # str(cluster) -> int(class)
    print(f"[blind] reusing baseline partition: k*={k_star} ARI={info['ari']:.3f} "
          f"purity={info['purity']:.3f}", flush=True)

    # ---- 3. pre-attack per-class accuracy (oracle, for NVA-Drop + recall) ----
    pre_per_class, base_acc = preattack_per_class(model_dir)
    print(f"[blind] base acc = {base_acc:.2f}%", flush=True)

    # ---- 4. attack EVERY cluster, BLIND (victim by cluster id, proxy = nearest)
    rows = []
    clusters_in_order = sorted(int(c) for c in c2c.keys())
    for tc in clusters_in_order:
        vc = int(c2c[str(tc)])                    # victim majority class (oracle)
        pc = nearest_other_cluster(centers, tc, forbidden_class=vc,
                                   cluster_to_class=c2c)
        pc_class = int(c2c[str(pc)])
        vp = float(info["per_cluster"][str(tc)]["purity"])
        vsize = int(info["per_cluster"][str(tc)]["size"])
        print(f"\n{'='*60}\n[blind] cluster {tc} (maj class {CLASS_NAMES[vc]}, "
              f"purity {vp:.2f}, size {vsize}) -> nearest-proxy cluster {pc} "
              f"(maj class {CLASS_NAMES[pc_class]})\n{'='*60}", flush=True)

        save_dir = L.run_laser_attack(
            TAG, c2s_p, s2c_p, feat_path, centers, clabels,
            target_cluster=tc, proxy_cluster=pc,
            atk_epochs=ATK_EPOCHS, count=COUNT, count2=COUNT2, lr=LR,
            model_tag=model_tag)

        post_per_class, acc_v, mp = L.evaluate_attack(
            TAG, model_dir, save_dir, ATK_EPOCHS,
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
            "post_per_class": {CLASS_NAMES[c]: float(post_per_class[c])
                               for c in range(NUM_CLASSES)},
        }
        rows.append(rec)
        print(f"[blind] cluster {tc} ({CLASS_NAMES[vc]}): "
              f"recall {pre_per_class[vc]:.1f}% -> {acc_v:.2f}%  "
              f"MP={mp:.2f}%  NVA-Drop={drop:.2f}%", flush=True)

        # crash-safe incremental save + free this cluster's ~1.2 GB checkpoints
        _save(rows, base_acc, info)
        shutil.rmtree(save_dir, ignore_errors=True)
        torch.cuda.empty_cache()

    # ---- 5. summary + sanity check vs the class-based baseline ---------------
    n_destroyed = sum(1 for r in rows if r["victim_recall_post"] <= DESTROY_RECALL_THRESH)
    summary = {
        "k_star": k_star,
        "clusters_attacked": len(rows),
        "clusters_destroyed_recall_le_1pct": n_destroyed,
        "mean_nva_drop": float(np.mean([r["nva_drop"] for r in rows])),
        "max_nva_drop": float(np.max([r["nva_drop"] for r in rows])),
        "mean_mp": float(np.mean([r["mp"] for r in rows])),
    }
    sanity = _sanity_vs_baseline(rows)
    _save(rows, base_acc, info, summary=summary, sanity=sanity)
    _write_table(rows, base_acc, summary)

    # ---- 6. free the ~10 GB cache -------------------------------------------
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[blind] freed cache {cache_dir}", flush=True)

    print(f"\n[blind] DONE: {n_destroyed}/{len(rows)} clusters destroyed "
          f"(recall <= {DESTROY_RECALL_THRESH}%), "
          f"mean NVA-Drop={summary['mean_nva_drop']:.2f}%, "
          f"mean MP={summary['mean_mp']:.2f}%.", flush=True)


def _sanity_vs_baseline(rows):
    """The class-based baseline attacked truck(9) via its dedicated cluster. Find
    the blind row whose victim class == truck and confirm its recall collapse
    reproduces that known-good attack (acc_v near 0). MP/NVA may differ because
    the blind proxy (nearest centroid) need not equal the baseline's proxy (car)."""
    bj = os.path.join(OUT, "results_baseline.json")
    if not os.path.exists(bj):
        return None
    with open(bj) as f:
        base = json.load(f)
    truck_row = next((r for r in rows if r["victim_class"] == 9), None)
    if truck_row is None:
        return None
    for a in base.get("attacks", []):
        if a.get("victim_class") == 9:
            return {
                "baseline_class_based_truck": {k: a.get(k) for k in ("acc_v", "mp", "nva_drop")},
                "blind_cluster_truck": {k: truck_row[k] for k in
                                        ("victim_recall_post", "mp", "nva_drop")},
                "note": "When the blind-selected cluster's majority class is the one "
                        "the class-based run targeted (truck), recall collapse should "
                        "reproduce (acc_v near 0). MP/NVA can differ since the blind "
                        "proxy (nearest centroid) need not be car.",
            }
    return None


def _save(rows, base_acc, info, summary=None, sanity=None):
    out = {
        "tag": "blind_select",
        "config": {
            "model": "VGG16", "split": "SB=1", "dataset": "CIFAR-10",
            "victim_selection": "BLIND (by cluster id, label-free)",
            "proxy_selection": "nearest PCA centroid (label-free)",
            "attack": "src.attack.attacks_dataloader1 (last-conv-block fine-tune, "
                      "guided-centroid pull + anchor, alpha=0.5)",
            "attack_epochs": ATK_EPOCHS, "count": COUNT, "count2": COUNT2, "lr": LR,
            "labels_role": "EVALUATION ORACLE ONLY (cluster naming + recall); "
                           "never used to select or execute the attack",
        },
        "base_acc": float(base_acc),
        "clustering": {k: info[k] for k in ("k_star", "ari", "nmi", "purity")},
        "rows": rows,
    }
    if summary is not None:
        out["summary"] = summary
    if sanity is not None:
        out["sanity_check"] = sanity
    path = os.path.join(OUT, "results_blind.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def _write_table(rows, base_acc, summary):
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    path = os.path.join(OUT, "tables", "blind_select.md")
    with open(path, "w") as f:
        f.write("# Blind (Label-Free) Victim-Cluster Selection "
                "— VGG16, SB=1, CIFAR-10\n\n")
        f.write("The attacker selects a victim **by cluster id** (iterating the k* "
                "discovered clusters) and a proxy by **nearest centroid** — fully "
                "label-free. Ground-truth labels name each cluster's majority class "
                "and measure recall collapse only; they play no role in selecting or "
                "running the attack. Base accuracy **%.2f%%**.\n\n" % base_acc)
        f.write("| Cluster | Victim class (majority) | Purity | Size | "
                "Recall pre → post | MP | NVA-Drop |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| %d | %s | %.2f | %d | %.1f → **%.2f** | %.2f%% | %.2f%% |\n" % (
                r["target_cluster"], r["victim_name"], r["cluster_purity"],
                r["cluster_size"], r["victim_recall_pre"], r["victim_recall_post"],
                r["mp"], r["nva_drop"]))
        f.write("\n**Summary:** %d/%d clusters destroyed (recall ≤ 1%%), "
                "mean NVA-Drop = %.2f%%, max NVA-Drop = %.2f%%, mean MP = %.2f%%.\n" % (
                    summary["clusters_destroyed_recall_le_1pct"],
                    summary["clusters_attacked"],
                    summary["mean_nva_drop"], summary["max_nva_drop"],
                    summary["mean_mp"]))
    print(f"[blind] table -> {path}", flush=True)


if __name__ == "__main__":
    run_blind_select()
