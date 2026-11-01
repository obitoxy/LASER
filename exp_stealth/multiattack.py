"""
multiattack.py — Cumulative multi-class LASER attack (RAID 296B follow-up).

Reviewer-relevant question: does attacking SEVERAL classes (instead of one) make
LASER more visible? We chain attacks cumulatively on the SAME server
(clean -> +cat -> +truck -> +bird) and evaluate after each step.

The expected — and honest — result: on the balanced monitoring set, each added
victim contributes ≈ (its prevalence × its accuracy) to Δglobal, so visibility
grows with the TOTAL victimized fraction (≈ #attacked × 10 % here). The per-class
audit also catches every collapsed class. This is the SAME prevalence mechanism
as the single-victim sweep (prevalence.py), not a new one — attacking more
classes simply raises the victimized fraction.

No change to the paper's attack: each step is the exact `attacks_dataloader1`,
just started from the previous step's server_20.pt (see run_laser_attack_from_ckpt).
"""
import os
import json
import time
import shutil
import numpy as np

import exp_robust.lib as L
from exp_robust.lib import OUT as ROUT, CLASS_NAMES
from exp_robust.experiments import find_cluster_for_class, pick_proxy_cluster
import exp_stealth.lib as S
from exp_stealth.lib import OUT as SOUT, MODEL_DIR

TAG = "multi"
ATK_EPOCHS = 20
CHAIN = [3, 9, 2]                       # cat -> truck -> bird (cumulative)


def run():
    t0 = time.time()
    loader = S.test_loader()

    # 1. cache + cluster (baseline model, label-free)
    print("[multi] caching embeddings (tag=multi) ...", flush=True)
    xtr, ytr, _, _ = L._base_cifar()
    c2s_p, s2c_p, lab_p = L.collect_embeddings(TAG, xtr, ytr, MODEL_DIR)
    clabels, centers, feat_path, info = L.cluster_features(TAG, s2c_p, lab_p, max_k=12)
    print(f"[multi] K*={info['k_star']} ARI={info['ari']:.3f} purity={info['purity']:.3f}",
          flush=True)

    # 2. clean reference outputs (computed once)
    print("[multi] forwarding clean reference ...", flush=True)
    clean_arrays = S.eval_clean(loader)

    # 3. cumulative chain
    ckpt = os.path.join(MODEL_DIR, "server_nor_last.pt")   # start from clean
    attacked_so_far = []
    steps = []
    for vic in CHAIN:
        tc = find_cluster_for_class(info["cluster_to_class"], vic)
        if tc is None:
            print(f"[multi] {CLASS_NAMES[vic]} has no dedicated cluster, skip", flush=True)
            continue
        pc_cluster, proxy_class = pick_proxy_cluster(info, tc, vic)
        print(f"[multi] +{CLASS_NAMES[vic]} (cluster {tc}) -> proxy "
              f"{CLASS_NAMES[proxy_class]} (cluster {pc_cluster}) | start={os.path.basename(os.path.dirname(ckpt))}",
              flush=True)
        step_dir = os.path.join(ROUT, "attacks", f"{TAG}_chain_v{vic}")
        ckpt = S.run_laser_attack_from_ckpt(
            c2s_p, s2c_p, feat_path, centers, clabels, tc, pc_cluster,
            start_ckpt=ckpt, save_dir=step_dir, atk_epochs=ATK_EPOCHS)
        attacked_so_far.append(vic)

        atk_arrays = S.eval_server(ckpt, loader)
        cmp = S.compare_to_clean(clean_arrays, atk_arrays, attacked=attacked_so_far)
        cmp.update({"step": len(attacked_so_far),
                    "attacked_classes": [CLASS_NAMES[c] for c in attacked_so_far],
                    "attacked_class_ids": list(attacked_so_far),
                    "new_victim": CLASS_NAMES[vic],
                    "proxy_for_new": CLASS_NAMES[proxy_class]})
        steps.append(cmp)
        pc_post = cmp["per_class_acc_post"]
        collapsed = [cn for cn in CLASS_NAMES if pc_post[cn] < 5.0]
        print(f"[multi] after {len(attacked_so_far)} class(es) {attacked_so_far}: "
              f"global {cmp['global_acc_pre']:.2f}->{cmp['global_acc_post']:.2f} "
              f"(Δ{cmp['global_acc_drop']:+.2f}%) | PSI={cmp['predicted_psi']:.2f} | "
              f"collapsed({len(collapsed)})={collapsed} | "
              f"non-atk conf Wass={cmp['conf_nonattacked']['wasserstein']:.3f} "
              f"shift={cmp['conf_nonattacked']['mean_shift']:+.3f}",
              flush=True)

    result = {"tag": TAG, "chain": [CLASS_NAMES[c] for c in CHAIN],
              "clustering": {k: info[k] for k in ("k_star", "ari", "purity")},
              "steps": steps, "runtime_sec": round(time.time() - t0, 1)}
    out = os.path.join(SOUT, "multiattack.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[multi] results -> {out}", flush=True)

    # 4. free the ~10 GB cache + chain checkpoints
    for d in (os.path.join(ROUT, "cache", TAG), os.path.join(ROUT, "clusters", TAG)):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    atk_root = os.path.join(ROUT, "attacks")
    if os.path.isdir(atk_root):
        for d in os.listdir(atk_root):
            if d.startswith(f"{TAG}_"):
                shutil.rmtree(os.path.join(atk_root, d), ignore_errors=True)
    print("[multi] freed cache + chain checkpoints", flush=True)
    print(f"[multi] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    run()
