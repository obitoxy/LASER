"""
cifar100_split.py — M7: CIFAR-100 split-depth study (VGG-16, SB=1/2/3).

Reuses `exp_scale.lib` (CIFAR-100, max_k=120, garbage-blob guard) but
parameterizes the split block.  `exp_scale.lib` was written for SB=1; its
`create_models()` is monkey-patched here to honor the active split.  All other
functions (training, caching, PCA/silhouette clustering, LASER attack,
evaluation) are called unchanged, so the pipeline is identical to the existing
CIFAR-100 result and only the cut changes.

Usage
-----
    python -m exp_camera.cifar100_split train --sb 2 --seed 9202
    python -m exp_camera.cifar100_split train --sb 3 --seed 9202
    python -m exp_camera.cifar100_split run   --sb 2 --seed 9202
    python -m exp_camera.cifar100_split run   --sb 3 --seed 9202
    python -m exp_camera.cifar100_split batch  # train + run SB=2,3 (seed 9202)

The SB=1 CIFAR-100 result already exists in exp_scale/outputs and is reused as
the Shallow row; only SB=2/3 need new training.  Single seed (9202) is used for
the two new cuts; this is stated in the paper table.  The blind protocol matches
exp_scale: 12 clusters selected by size-rank stratification, proxy by nearest
PCA centroid, garbage blobs excluded label-free, 10 victim + 50/non-victim
cached vectors, 20 epochs.
"""
import argparse
import json
import os
import random
import shutil
import time

import numpy as np
import torch

import exp_scale.lib as L

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT100 = os.path.join(REPO, "exp_camera", "outputs", "cifar100")
os.makedirs(OUT100, exist_ok=True)

# Redirect exp_scale.lib's artefacts to exp_camera/outputs/cifar100.
L.OUT = OUT100

ACTIVE_SB = 1
_ORIG_CREATE = L.create_models


def _create_models(sb=None):
    """Honor the active split for every exp_scale.lib function."""
    return _ORIG_CREATE(ACTIVE_SB if sb is None else sb)


L.create_models = _create_models

EPOCHS = 120
ATK_EPOCHS = 20
COUNT = 10
COUNT2 = 50
LR = 1e-4
N_BLIND = 12
DESTROY_THRESH = 1.0
SEEDS = (9202,)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def tag_for(sb, seed):
    return f"cifar100_sb{sb}_seed{seed}"


def model_dir(sb, seed):
    return os.path.join(OUT100, "models", tag_for(sb, seed))


def _pc(info, cluster_id):
    """per_cluster record with either str or int keys."""
    per = info["per_cluster"]
    return per.get(str(cluster_id), per.get(cluster_id))


def _c2c(info, cluster_id):
    c2c = info["cluster_to_class"]
    return int(c2c.get(str(cluster_id), c2c.get(cluster_id)))


def _select_blind_clusters(info, n, exclude=None):
    """Same label-free size-rank stratification as exp_scale.main."""
    exclude = exclude or set()
    pcs = info["per_cluster"]
    sizes = sorted(((int(c), pcs[str(c)]["size"]) for c in pcs
                    if int(c) not in exclude), key=lambda x: x[1])
    if len(sizes) <= n:
        sel = [c for c, _ in sizes]
    else:
        idx = np.linspace(0, len(sizes) - 1, n).round().astype(int)
        idx = sorted(set(idx.tolist()))
        sel = [sizes[i][0] for i in idx]
    print(f"[M7] blind-selected {len(sel)} clusters by size-rank stratification "
          f"(label-free); sizes={[dict(sizes)[c] for c in sel]}", flush=True)
    return sel


# ======================================================================
# Train
# ======================================================================
def train(sb, seed=9202, epochs=EPOCHS):
    global ACTIVE_SB
    ACTIVE_SB = int(sb)
    set_seed(seed)
    tag = tag_for(sb, seed)
    xtr, ytr, _, _ = L._base_cifar100()
    print(f"=== M7 TRAIN {tag} (sb={sb}, seed={seed}, epochs={epochs}) ===",
          flush=True)
    save_dir, best = L.train_split_model(tag, xtr, ytr, epochs=epochs)
    config = {
        "tag": tag, "dataset": "CIFAR-100", "architecture": "VGG-16",
        "split_block": int(sb), "seed": int(seed), "epochs": int(epochs),
        "num_classes": int(L.NUM_CLASSES), "train_size": int(len(xtr)),
        "protocol": "same as exp_scale CIFAR-100 (SGD+mixup, 120 epochs)",
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    hist_path = os.path.join(save_dir, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist = json.load(f)
        hist["config"] = config
        with open(hist_path, "w") as f:
            json.dump(hist, f, indent=2)
    print(f"[M7] DONE train {tag}: best_acc={best:.2f}%", flush=True)
    return save_dir, best


# ======================================================================
# Cache -> cluster -> blind attacks -> eval
# ======================================================================
def run(sb, seed=9202, n_blind=N_BLIND, atk_epochs=ATK_EPOCHS):
    global ACTIVE_SB
    ACTIVE_SB = int(sb)
    set_seed(seed)
    tag = tag_for(sb, seed)
    mdir = model_dir(sb, seed)
    assert os.path.exists(os.path.join(mdir, "server_nor_last.pt")), \
        f"model not trained: {mdir}; run train --sb {sb} --seed {seed} first"

    result_path = os.path.join(OUT100, f"results_{tag}.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            old = json.load(f)
        if old.get("status") == "done":
            print(f"[M7] skip {tag}: {result_path} exists", flush=True)
            return old

    xtr, ytr, _, _ = L._base_cifar100()
    print(f"\n=== M7 RUN {tag} ===", flush=True)
    c2s_p, s2c_p, lab_p = L.collect_embeddings(tag, xtr, ytr, mdir)

    clabels, centers, feat_path, info_raw = L.cluster_features(
        tag, s2c_p, lab_p, max_k=L.MAX_K)
    # Reload JSON so per_cluster / cluster_to_class keys are strings, matching
    # exp_scale.main and the helper functions that expect str keys.
    cluster_dir = os.path.join(OUT100, "clusters", tag)
    with open(os.path.join(cluster_dir, "info.json")) as f:
        info = json.load(f)
    clabels = np.load(os.path.join(cluster_dir, "cluster_labels.npy")).astype(np.int64)
    centers = np.load(os.path.join(cluster_dir, "centers_pca.npy"))

    pre_per_class, base_acc = L.preattack_per_class(mdir)
    print(f"[M7] {tag}: base acc={base_acc:.2f}% k*={info['k_star']} "
          f"ARI={info['ari']:.3f} purity={info['purity']:.3f}", flush=True)

    blobs = L.blob_cluster_ids(info)
    if blobs:
        print(f"[M7] blob guard excludes {sorted(blobs)}", flush=True)
    selected = _select_blind_clusters(info, n_blind, exclude=blobs)
    rows = []
    rec = {
        "status": "running", "tag": tag, "dataset": "CIFAR-100",
        "architecture": "VGG-16", "split_block": int(sb), "seed": int(seed),
        "config": {
            "max_k": int(L.MAX_K), "n_blind": int(n_blind),
            "atk_epochs": int(atk_epochs), "count": COUNT, "count2": COUNT2,
            "lr": LR, "victim_selection": "BLIND size-rank stratified; "
            "garbage blobs excluded", "proxy_selection": "nearest PCA centroid",
            "labels_role": "evaluation oracle only",
        },
        "clustering": {k: info[k] for k in
                       ("k_star", "max_k", "ari", "nmi", "purity")},
        "base_acc": float(base_acc),
        "rows": rows,
    }
    _save(rec, result_path)

    for tc in selected:
        vc = _c2c(info, tc)
        vp = float(_pc(info, tc)["purity"])
        vsize = int(_pc(info, tc)["size"])
        pc = L.nearest_other_cluster(
            centers, tc, forbidden_class=vc,
            cluster_to_class=info["cluster_to_class"], forbidden_clusters=blobs)
        pc_class = _c2c(info, pc)
        print(f"[M7] cluster {tc} ({L.CLASS_NAMES[vc]}, purity {vp:.2f}, "
              f"size {vsize}) -> proxy {pc} ({L.CLASS_NAMES[pc_class]})",
              flush=True)
        atk_dir = L.run_laser_attack(
            tag, c2s_p, s2c_p, feat_path, centers, clabels,
            target_cluster=tc, proxy_cluster=pc,
            atk_epochs=atk_epochs, count=COUNT, count2=COUNT2, lr=LR,
            model_tag=tag)
        post_per_class, acc_v, mp = L.evaluate_attack(
            tag, mdir, atk_dir, atk_epochs, vc, pc_class)
        drop, pre_nv, post_nv = L.nva_drop(pre_per_class, post_per_class, vc)
        row = {
            "target_cluster": int(tc), "victim_class": int(vc),
            "victim_name": L.CLASS_NAMES[vc], "proxy_cluster": int(pc),
            "proxy_class": int(pc_class), "proxy_name": L.CLASS_NAMES[pc_class],
            "cluster_purity": vp, "cluster_size": vsize,
            "victim_recall_pre": float(pre_per_class[vc]),
            "victim_recall_post": float(acc_v), "mp": float(mp),
            "nva_drop": float(drop),
            "nonvictim_acc_pre": float(pre_nv),
            "nonvictim_acc_post": float(post_nv),
        }
        rows.append(row)
        print(f"[M7] {tag} {L.CLASS_NAMES[vc]}: {pre_per_class[vc]:.1f}% -> "
              f"{acc_v:.2f}%  MP={mp:.2f}%  NVA={drop:.2f}%", flush=True)
        _save(rec, result_path)
        shutil.rmtree(atk_dir, ignore_errors=True)
        torch.cuda.empty_cache()

    n_destroyed = sum(1 for r in rows if r["victim_recall_post"] <= DESTROY_THRESH)
    rec["status"] = "done"
    rec["summary"] = {
        "clusters_attacked": len(rows),
        "clusters_destroyed_recall_le_1pct": n_destroyed,
        "mean_nva_drop": float(np.mean([r["nva_drop"] for r in rows])) if rows else 0.0,
        "max_nva_drop": float(np.max([r["nva_drop"] for r in rows])) if rows else 0.0,
        "mean_mp": float(np.mean([r["mp"] for r in rows])) if rows else 0.0,
        "n_garbage_blobs_excluded": len(blobs),
    }
    _save(rec, result_path)
    _write_table(rec, result_path.replace("results_", "tables/").replace(".json", ".md"))
    shutil.rmtree(os.path.join(OUT100, "cache", tag), ignore_errors=True)
    free_gb = shutil.disk_usage("/root/autodl-tmp").free / 1e9
    print(f"[M7] DONE {tag}: {n_destroyed}/{len(rows)} destroyed; "
          f"free={free_gb:.1f} GB", flush=True)
    return rec


def _save(rec, path):
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)


def _write_table(rec, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# M7 — CIFAR-100 split {rec['split_block']} (VGG-16)\n\n")
        f.write(f"Base Acc **{rec['base_acc']:.2f}%**; K*={rec['clustering']['k_star']} "
                f"(true 100), ARI={rec['clustering']['ari']:.3f}, "
                f"NMI={rec['clustering']['nmi']:.3f}, "
                f"purity={rec['clustering']['purity']:.3f}.  \n")
        f.write("Blind label-free victim selection (12 clusters, size-rank "
                "stratified), proxy = nearest PCA centroid, garbage blobs "
                "excluded. 10 victim + 50/non-victim cached vectors, 20 epochs.\n\n")
        f.write("| Cluster | Victim | Purity | Size | Recall pre → post | MP | NVA-Drop |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in rec["rows"]:
            f.write(f"| {r['target_cluster']} | {r['victim_name']} | "
                    f"{r['cluster_purity']:.2f} | {r['cluster_size']} | "
                    f"{r['victim_recall_pre']:.1f} → **{r['victim_recall_post']:.2f}** | "
                    f"{r['mp']:.2f}% | {r['nva_drop']:.2f}% |\n")
        s = rec.get("summary", {})
        if s:
            f.write(f"\n**Summary:** {s['clusters_destroyed_recall_le_1pct']}/"
                    f"{s['clusters_attacked']} destroyed (recall ≤1%), mean MP "
                    f"{s['mean_mp']:.2f}%, mean NVA-Drop {s['mean_nva_drop']:.2f}%, "
                    f"max NVA-Drop {s['max_nva_drop']:.2f}%.\n")


def batch(epochs=EPOCHS, n_blind=N_BLIND, atk_epochs=ATK_EPOCHS):
    for sb in (2, 3):
        train(sb, seed=9202, epochs=epochs)
        run(sb, seed=9202, n_blind=n_blind, atk_epochs=atk_epochs)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "run"):
        q = sub.add_parser(name)
        q.add_argument("--sb", type=int, required=True, choices=(1, 2, 3))
        q.add_argument("--seed", type=int, default=9202)
        q.add_argument("--epochs", type=int, default=EPOCHS)
        if name == "run":
            q.add_argument("--n-blind", type=int, default=N_BLIND)
            q.add_argument("--atk-epochs", type=int, default=ATK_EPOCHS)
    qb = sub.add_parser("batch")
    qb.add_argument("--epochs", type=int, default=EPOCHS)
    qb.add_argument("--n-blind", type=int, default=N_BLIND)
    qb.add_argument("--atk-epochs", type=int, default=ATK_EPOCHS)
    args = p.parse_args()
    if args.cmd == "train":
        train(args.sb, seed=args.seed, epochs=args.epochs)
    elif args.cmd == "run":
        run(args.sb, seed=args.seed, n_blind=args.n_blind,
            atk_epochs=args.atk_epochs)
    else:
        batch(epochs=args.epochs, n_blind=args.n_blind,
              atk_epochs=args.atk_epochs)


if __name__ == "__main__":
    main()
