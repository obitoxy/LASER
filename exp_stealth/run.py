"""
run.py — Stealth / client-detection experiment driver (RAID 2026, reviewer 296B).

Pipeline (all reusing the exact LASER code path from exp_robust / src):
    1. cache smashed-data + server features for the baseline clean model
       (VGG-16, SB=1, CIFAR-10) -- the same model the paper's main table uses;
    2. unsupervised silhouette K-search + K-means clustering (label-free);
    3. for each victim class: pick a proxy cluster, run the LASER attack
       (last-conv-block fine-tune, guided centroid + anchoring, 20 epochs) and
       evaluate the attacked server_20 against the clean client;
    4. Part A (validation monitoring) + Part B (prediction drift) stealth eval;
    5. write outputs/results.json + outputs/eval/<victim>.npz (raw arrays for
       the figure script), then FREE the ~10 GB embedding cache and the attack
       checkpoints (disk hygiene, see CLAUDE.md).

The clean model is NOT retrained — it is reused from exp_robust/outputs/models/
baseline so the stealth numbers sit on top of the exact model that reproduces
the paper's headline table.

Usage (conda env `cdssc`, from repo root):
    python -m exp_stealth.run             # full: cache -> cluster -> attack -> eval
    python -m exp_stealth.run --eval-only # skip attack, re-eval existing ckpts
"""
import os
import sys
import json
import time
import shutil
import argparse
import numpy as np
import torch

import exp_robust.lib as L
from exp_robust.lib import OUT as ROUT, CLASS_NAMES
from exp_robust.experiments import find_cluster_for_class, pick_proxy_cluster
import exp_stealth.lib as S
from exp_stealth.lib import OUT as SOUT, MODEL_DIR

TAG = "stealth"                       # cache/cluster/attack tag (isolated from exp_robust)
ATK_EPOCHS = 20
VICTIMS = [3, 9]                      # cat (hardest) and truck (matches paper baseline)


def cache_and_cluster():
    """Cache the baseline model's train embeddings (label-free server side) and
    cluster them. Returns (c2s_path, s2c_path, feat_path, centers, clabels, info)."""
    print("\n[stealth] caching baseline embeddings (tag=stealth) ...", flush=True)
    xtr, ytr, _, _ = L._base_cifar()
    c2s_p, s2c_p, lab_p = L.collect_embeddings(TAG, xtr, ytr, MODEL_DIR)
    print("[stealth] clustering ...", flush=True)
    clabels, centers, feat_path, info = L.cluster_features(TAG, s2c_p, lab_p, max_k=12)
    print(f"[stealth] K*={info['k_star']} ARI={info['ari']:.3f} "
          f"purity={info['purity']:.3f}", flush=True)
    return c2s_p, s2c_p, feat_path, centers, clabels, info


def run_one_victim(victim, c2s_p, s2c_p, feat_path, centers, clabels, info,
                   loader, atk_dir_of, skip_attack):
    """Attack one victim class + run the stealth eval. Returns (record, paths)."""
    tc = find_cluster_for_class(info["cluster_to_class"], victim)
    if tc is None:
        print(f"[stealth] victim={CLASS_NAMES[victim]} has NO dedicated cluster -> "
              f"cannot target label-free", flush=True)
        return None
    pc_cluster, proxy_class = pick_proxy_cluster(info, tc, victim)
    print(f"\n[stealth] VICTIM={CLASS_NAMES[victim]} (cluster {tc}) -> "
          f"proxy={CLASS_NAMES[proxy_class]} (cluster {pc_cluster})", flush=True)

    atk_dir = atk_dir_of(victim, tc, pc_cluster)
    ckpt = os.path.join(atk_dir, f"server_{ATK_EPOCHS}.pt")
    if not skip_attack:
        L.run_laser_attack(TAG, c2s_p, s2c_p, feat_path, centers, clabels,
                           target_cluster=tc, proxy_cluster=pc_cluster,
                           atk_epochs=ATK_EPOCHS, count=10, count2=50,
                           suffix=f"_v{victim}", model_tag="baseline")
    else:
        assert os.path.exists(ckpt), f"--eval-only but {ckpt} missing"

    t0 = time.time()
    res = S.evaluate_stealth(victim=victim, proxy=proxy_class, server_ckpt=ckpt,
                             loader=loader)
    print(f"[stealth] eval done in {time.time()-t0:.1f}s", flush=True)

    # persist the raw per-sample arrays (for the figure script) and the metric dict
    raw_path = os.path.join(SOUT, "eval", f"victim{victim}.npz")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    np.savez_compressed(raw_path,
                        probs_clean=res["probs_clean"], preds_clean=res["preds_clean"],
                        probs_atk=res["probs_atk"], preds_atk=res["preds_atk"],
                        conf_clean=res["conf_clean"], conf_atk=res["conf_atk"],
                        ent_clean=res["ent_clean"], ent_atk=res["ent_atk"],
                        labels=res["labels"])
    rec = {
        "victim": int(victim), "victim_name": CLASS_NAMES[victim],
        "proxy": int(proxy_class), "proxy_name": CLASS_NAMES[proxy_class],
        "target_cluster": int(tc), "proxy_cluster": int(pc_cluster),
        "server_ckpt": ckpt,
        "partA_monitoring": res["partA_monitoring"],
        "partB_drift": res["partB_drift"],
    }
    A = res["partA_monitoring"]; Bp = res["partB_drift"]
    print(f"[stealth] >>> {CLASS_NAMES[victim]}: global {A['global_acc_pre']:.2f}%->"
          f"{A['global_acc_post']:.2f}% (Δ{A['global_acc_abs_drop']:+.2f}%) | "
          f"ACC-V {A['acc_victim_pre']:.1f}->{A['acc_victim_post']:.1f}% | "
          f"MP {A['misclass_prob']:.1f}% | NVA-Drop {A['nva_drop']:.2f}% | "
          f"conf-KS(all) {Bp['confidence']['all']['ks_d']:.3f} "
          f"nonvictim {Bp['confidence']['nonvictim']['ks_d']:.3f} | "
          f"hist-PSI {Bp['predicted_class_hist']['psi']:.3f} "
          f"(null95={Bp['detection_thresholds']['psi_null_95pct_5pctFPR']:.3f}, "
          f"detected={Bp['detection_thresholds']['psi_attack_detected']})",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true",
                    help="skip caching/clustering/attack; re-eval existing ckpts")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="keep the embedding cache + attack checkpoints after eval")
    ap.add_argument("--victims", type=str, default=",".join(map(str, VICTIMS)))
    args = ap.parse_args()
    victims = [int(v) for v in args.victims.split(",") if v.strip() != ""]

    loader = S.test_loader()
    t_start = time.time()

    if args.eval_only:
        # need the cluster info only to recover victim->proxy mapping
        info_path = os.path.join(ROUT, "clusters", TAG, "info.json")
        with open(info_path) as f:
            info = json.load(f)
        c2s_p = s2c_p = feat_path = None
        centers = clabels = None
    else:
        c2s_p, s2c_p, feat_path, centers, clabels, info = cache_and_cluster()

    def atk_dir_of(victim, tc, pc):
        return os.path.join(ROUT, "attacks", f"{TAG}_t{tc}_p{pc}_v{victim}")

    results = {"tag": TAG, "model": "baseline (VGG-16, SB=1)",
               "dataset": "CIFAR-10", "attack_epochs": ATK_EPOCHS,
               "clustering": {k: info[k] for k in ("k_star", "ari", "nmi", "purity")},
               "victims": []}
    for v in victims:
        rec = run_one_victim(v, c2s_p, s2c_p, feat_path, centers, clabels, info,
                             loader, atk_dir_of, args.eval_only)
        if rec is not None:
            results["victims"].append(rec)

    results["runtime_sec"] = round(time.time() - t_start, 1)
    out_path = os.path.join(SOUT, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[stealth] results -> {out_path}", flush=True)

    # ---- disk hygiene: free the ~10 GB cache + attack checkpoints ----
    if not args.no_cleanup:
        cache_dir = os.path.join(ROUT, "cache", TAG)
        clus_dir = os.path.join(ROUT, "clusters", TAG)
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"[stealth] freed cache {cache_dir}", flush=True)
        # attack checkpoints are ~1.2 GB each; drop them now that eval is saved.
        # All dirs under attacks/ starting with our TAG belong to this experiment.
        atk_root = os.path.join(ROUT, "attacks")
        if os.path.isdir(atk_root):
            for d in os.listdir(atk_root):
                if d.startswith(f"{TAG}_"):
                    shutil.rmtree(os.path.join(atk_root, d), ignore_errors=True)
        print(f"[stealth] freed attack checkpoints under {ROUT}/attacks/{TAG}_*",
              flush=True)
        # keep clusters/<TAG> (small) so --eval-only can reuse the mapping

    print(f"[stealth] DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
