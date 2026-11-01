"""
experiments.py — Drivers for the three complex / non-ideal distribution
experiments requested by reviewers 296A & 296B, plus the clean baseline.

Each static condition (baseline / class-imbalance / label-noise) follows the
identical pipeline:
    trained model -> cache clean-label embeddings -> unsupervised clustering
    -> for each chosen (victim, proxy) class pair: LASER attack + evaluation.

The dynamic-distribution experiment is separate (run_dynamic).
"""
import os
import json
import numpy as np
import torch

import exp_robust.lib as L
from exp_robust.lib import OUT, NUM_CLASSES, CLASS_NAMES


# ----------------------------------------------------------------------
# Pre-attack (clean server) per-class accuracy — needed for NVA Drop.
# ----------------------------------------------------------------------
@torch.no_grad()
def preattack_per_class(model_dir):
    net1, server, net3 = L.create_models()
    net1.load_state_dict(torch.load(os.path.join(model_dir, "net1_nor_last.pt")))
    server.load_state_dict(torch.load(os.path.join(model_dir, "server_nor_last.pt")))
    net3.load_state_dict(torch.load(os.path.join(model_dir, "net3_nor_last.pt")))
    net1.eval(); server.eval(); net3.eval()
    import torchvision
    from torch.utils.data import DataLoader
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=L.get_transforms(False))
    loader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=8)
    per_correct = np.zeros(NUM_CLASSES); per_total = np.zeros(NUM_CLASSES)
    overall_c = overall_t = 0
    for x, y in loader:
        x, y = x.to(L.DEVICE), y.to(L.DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = net3(server(net1(x)))
        pred = out.argmax(1)
        overall_c += pred.eq(y).sum().item(); overall_t += y.size(0)
        for c in range(NUM_CLASSES):
            m = y == c
            per_total[c] += m.sum().item()
            per_correct[c] += (pred[m] == c).sum().item()
    return 100.0 * per_correct / np.maximum(per_total, 1), 100.0 * overall_c / overall_t


def find_cluster_for_class(cluster_to_class, target_class):
    """Return the cluster id whose majority true-class == target_class, else None."""
    for c, cls in cluster_to_class.items():
        if cls == target_class:
            return c
    return None


def pick_proxy_cluster(info, victim_cluster, victim_class):
    """Pick the nearest *other* cluster (by silhouette-cluster proximity heuristic:
    largest, most-pure cluster whose majority class != victim_class) as proxy."""
    best = None
    for c, cls in info["cluster_to_class"].items():
        if c == victim_cluster or cls == victim_class:
            continue
        pc = info["per_cluster"].get(c, {})
        score = (pc.get("purity", 0), pc.get("size", 0))
        if best is None or score > best[0]:
            best = (score, c, cls)
    return best[1], best[2]


def run_static_condition(tag, model_dir, cache_images, cache_labels_clean,
                          victim_classes, atk_epochs=20, atk_count=10,
                          atk_count2=50, max_k=12, min_cluster_purity=0.5):
    """Full pipeline for one static distribution condition.

    cache_images / cache_labels_clean : the TRAINING distribution the server
        caches smashed data from (its labels are CLEAN ground truth and used
        ONLY for cluster-quality measurement, never by the attack).
    victim_classes : list of TRUE class ids to attack.
    min_cluster_purity : a victim is only attacked if its dedicated cluster is
        at least this pure; otherwise the cluster is a blended blob and an
        attack on it would be indiscriminate (reported as IMPURE_CLUSTER).
    """
    print(f"\n{'='*70}\nCONDITION: {tag}\n{'='*70}", flush=True)
    results = {"tag": tag, "victim_classes": list(victim_classes), "attacks": []}

    # 1. cache embeddings
    c2s_p, s2c_p, lab_p = L.collect_embeddings(tag, cache_images, cache_labels_clean, model_dir)

    # 2. cluster
    clabels, centers, feat_path, info = L.cluster_features(tag, s2c_p, lab_p, max_k=max_k)
    results["clustering"] = {k: info[k] for k in
                             ("k_star", "ari", "nmi", "purity", "sil_curve",
                              "per_cluster")}
    # class coverage: which true classes have a dedicated majority cluster?
    covered = sorted(set(info["cluster_to_class"].values()))
    results["clustering"]["classes_with_dedicated_cluster"] = covered

    # 3. pre-attack per-class accuracy
    pre_per_class, base_acc = preattack_per_class(model_dir)
    results["base_acc"] = base_acc
    results["pre_per_class"] = {CLASS_NAMES[c]: float(pre_per_class[c]) for c in range(NUM_CLASSES)}

    # 4. for each victim class: attack + evaluate
    for vc in victim_classes:
        tc = find_cluster_for_class(info["cluster_to_class"], vc)
        if tc is None:
            # victim class has NO dedicated cluster -> attacker cannot target it
            results["attacks"].append({
                "victim_class": vc, "victim_name": CLASS_NAMES[vc],
                "status": "NO_DEDICATED_CLUSTER",
                "note": "victim class is absorbed into another cluster; the "
                        "label-free attacker cannot isolate it."
            })
            print(f"[{tag}] victim={CLASS_NAMES[vc]}: NO dedicated cluster -> "
                  f"cannot target label-free", flush=True)
            continue
        # purity gate: if the victim's cluster is a blended blob, attacking it
        # is indiscriminate rather than targeted -> report, do not attack.
        vp = info["per_cluster"].get(tc, {}).get("purity", 0.0)
        if vp < min_cluster_purity:
            results["attacks"].append({
                "victim_class": vc, "victim_name": CLASS_NAMES[vc],
                "status": "IMPURE_CLUSTER", "cluster_purity": float(vp),
                "note": f"victim's cluster purity {vp:.2f} < {min_cluster_purity}; "
                        "features collapsed, attack would be indiscriminate."
            })
            print(f"[{tag}] victim={CLASS_NAMES[vc]}: cluster purity {vp:.2f} "
                  f"< {min_cluster_purity} -> NOT cleanly targetable", flush=True)
            continue
        pc_cluster, proxy_class = pick_proxy_cluster(info, tc, vc)
        print(f"[{tag}] victim={CLASS_NAMES[vc]} (cluster {tc}) -> "
              f"proxy={CLASS_NAMES[proxy_class]} (cluster {pc_cluster})", flush=True)

        atk_dir = L.run_laser_attack(tag, c2s_p, s2c_p, feat_path, centers, clabels,
                                     target_cluster=tc, proxy_cluster=pc_cluster,
                                     atk_epochs=atk_epochs, count=atk_count,
                                     count2=atk_count2, suffix=f"_v{vc}")
        atk_epoch = atk_epochs  # use final epoch
        post_per_class, acc_v, mp = L.evaluate_attack(tag, model_dir, atk_dir, atk_epoch,
                                                      victim_class=vc, proxy_class=proxy_class)
        drop, pre_nv, post_nv = L.nva_drop(pre_per_class, post_per_class, vc)
        rec = {
            "victim_class": vc, "victim_name": CLASS_NAMES[vc],
            "proxy_class": proxy_class, "proxy_name": CLASS_NAMES[proxy_class],
            "target_cluster": int(tc), "proxy_cluster": int(pc_cluster),
            "acc_v": float(acc_v), "mp": float(mp),
            "nva_drop": float(drop),
            "nonvictim_acc_pre": float(pre_nv),
            "nonvictim_acc_post": float(post_nv),
            "post_per_class": {CLASS_NAMES[c]: float(post_per_class[c]) for c in range(NUM_CLASSES)},
        }
        results["attacks"].append(rec)
        print(f"[{tag}] VICTIM={CLASS_NAMES[vc]}  ACC-V={acc_v:.2f}%  "
              f"MP={mp:.2f}%  NVA-Drop={drop:.2f}%", flush=True)

    save_path = os.path.join(OUT, f"results_{tag}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{tag}] results -> {save_path}", flush=True)

    # free the large feature caches now that attacks + eval are done (c2s/s2c are
    # ~10 GB together and are not needed downstream; clusters + models are kept).
    import shutil
    cache_dir = os.path.join(OUT, "cache", tag)
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[{tag}] freed cache {cache_dir}", flush=True)
    return results
