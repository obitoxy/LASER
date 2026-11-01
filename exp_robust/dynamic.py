"""
dynamic.py — Dynamic (non-stationary) data distribution experiment.

Motivation (reviewer 296B): "the evaluation should more directly test whether
this behavior persists under dynamic data distributions." In a real USL
deployment the client's data is a stream whose class composition changes over
time. The server keeps caching smashed data and periodically re-clusters and
re-attacks.

Design — three temporal phases of the client stream (using the BASELINE clean
model, no retraining):
    Phase A (t1): client stream = classes {0,1,2,3,4}            (5 active classes)
    Phase B (t2): client stream = all 10 classes                 (10 active classes)
    Phase C (t3): client stream = classes {5,6,7,8,9}            (5 active classes)

For each phase the server:
    1. caches that phase's smashed data (a class-filtered train subset),
    2. runs the *unsupervised* silhouette K-search + clustering  (no labels!),
    3. executes LASER on a victim class active in that phase,
    4. is evaluated on the phase's test subset.

Claim: because LASER re-clusters from scratch each phase, its K-search TRACKS
the changing number of active classes (5 -> 10 -> 5) and the attack stays
effective phase after phase. This shows LASER is robust to dynamic data.
"""
import os
import json
import numpy as np

import exp_robust.lib as L
from exp_robust.lib import OUT, CLASS_NAMES
from exp_robust.experiments import (preattack_per_class, find_cluster_for_class,
                                    pick_proxy_cluster)


def phase_indices(train_labels, classes, per_class=5000, seed=0):
    rng = np.random.RandomState(seed)
    idx = []
    for c in classes:
        ic = np.where(train_labels == c)[0]
        rng.shuffle(ic)
        idx.extend(ic[:per_class].tolist())
    return np.array(sorted(idx))


def run_phase(phase_tag, model_tag, model_dir, train_images, train_labels,
              active_classes, victim_class, base_pre_per_class, per_class_n=5000):
    active = list(active_classes)
    idx = phase_indices(train_labels, active, per_class=per_class_n)
    cache_imgs = train_images[idx]
    cache_labels = train_labels[idx]

    # cache phase embeddings
    c2s_p, s2c_p, lab_p = L.collect_embeddings(
        phase_tag, cache_imgs, cache_labels, model_dir, cache_suffix=f"_{phase_tag}")

    # cluster
    clabels, centers, feat_path, info = L.cluster_features(phase_tag + "_cls", s2c_p, lab_p, max_k=12)

    # victim must be active & have a dedicated cluster
    tc = find_cluster_for_class(info["cluster_to_class"], victim_class)
    if tc is None:
        return {"phase": phase_tag, "active_classes": active,
                "k_star": info["k_star"], "ari": info["ari"], "purity": info["purity"],
                "victim_class": victim_class, "status": "NO_DEDICATED_CLUSTER"}
    pc_cluster, proxy_class = pick_proxy_cluster(info, tc, victim_class)

    atk_dir = L.run_laser_attack(phase_tag, c2s_p, s2c_p, feat_path, centers, clabels,
                                 target_cluster=tc, proxy_cluster=pc_cluster,
                                 atk_epochs=20, count=10, count2=50,
                                 suffix=f"_{phase_tag}", model_tag=model_tag)
    post_per_class, acc_v, mp = L.evaluate_attack(
        phase_tag, model_dir, atk_dir, 20, victim_class, proxy_class,
        test_classes=active)
    # NVA drop within the active (non-victim) classes of this phase
    others = [c for c in active if c != victim_class]
    pre_nv = float(np.mean([base_pre_per_class[c] for c in others]))
    post_nv = float(np.mean([post_per_class[c] for c in others]))
    drop = pre_nv - post_nv
    rec = {
        "phase": phase_tag, "active_classes": active,
        "true_n_active": len(active),
        "k_star": int(info["k_star"]), "ari": float(info["ari"]),
        "purity": float(info["purity"]),
        "victim_class": victim_class, "victim_name": CLASS_NAMES[victim_class],
        "proxy_class": int(proxy_class), "proxy_name": CLASS_NAMES[proxy_class],
        "acc_v": float(acc_v), "mp": float(mp),
        "nva_drop": float(drop),
        "nonvictim_acc_pre": pre_nv, "nonvictim_acc_post": post_nv,
    }
    print(f"[dynamic/{phase_tag}] active={active} k*={info['k_star']} "
          f"(true {len(active)}) VICTIM={CLASS_NAMES[victim_class]} "
          f"ACC-V={acc_v:.2f}% MP={mp:.2f}% NVA-Drop={drop:.2f}%", flush=True)

    # free this phase's large feature caches now that its attack + eval are done
    import shutil
    for d in (os.path.join(L.OUT, "cache", phase_tag),
              os.path.join(L.OUT, "clusters", phase_tag + "_cls")):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    return rec


def run_dynamic(model_tag="baseline"):
    from exp_robust.experiments import run_static_condition  # noqa (avoid dup import cost)
    print(f"\n{'='*70}\nCONDITION: dynamic distribution (model={model_tag})\n{'='*70}",
          flush=True)
    model_dir = os.path.join(OUT, "models", model_tag)
    xtr, ytr, _, _ = L._base_cifar()
    base_pre_per_class, base_acc = preattack_per_class(model_dir)

    phases = [
        ("phaseA_classes0to4", list(range(0, 5)), 1),   # victim = car
        ("phaseB_all10",       list(range(0, 10)), 9),  # victim = truck
        ("phaseC_classes5to9", list(range(5, 10)), 8),  # victim = ship
    ]
    phase_results = []
    for ptag, classes, victim in phases:
        r = run_phase(ptag, model_tag, model_dir, xtr, ytr,
                      active_classes=classes, victim_class=victim,
                      base_pre_per_class=base_pre_per_class)
        phase_results.append(r)

    summary = {"model_tag": model_tag, "base_acc": base_acc, "phases": phase_results}
    save_path = os.path.join(OUT, "results_dynamic.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[dynamic] results -> {save_path}", flush=True)
    return summary
