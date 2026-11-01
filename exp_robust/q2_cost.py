"""
q2_cost.py — LASER attack computational OVERHEAD + resource-constraint SWEEP.

Answers RAID 2026 reviewer 296B #5: "the computational overhead of the attack and
the trade-off between attack success, stealth, and resource cost", including
"edge environments with limited computational resources."

Two measurements in one run (the ~10 GB embedding cache is built once, then freed):

  (1) OVERHEAD — parameter budget (the trainable FRACTION of the server f_s),
      plus wall-clock and peak GPU memory of (a) the one-time embedding cache
      build and (b) one LASER fine-tune, benchmarked against one normal training
      epoch. The overhead is STRUCTURAL (last-conv-block fine-tune on a tiny
      cached subset), so the trainable-parameter FRACTION — not absolute FLOPs —
      is the architecture-portable headline number.

  (2) RESOURCE SWEEP — vary the victim cached-sample count and the attack epochs;
      show ACC-V -> 0 persists down to extreme low-resource settings (few samples,
      few epochs), supporting the "limited-resource / edge" claim.

VGG16 / SB=1 / CIFAR-10 only (the model on this server). Reuses models/baseline +
clusters/baseline; regenerates the cache once under cache/q2 (freed at the end).
Each attack's checkpoints are freed right after its evaluation (flat disk).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import shutil
import time
import numpy as np
import torch

import exp_robust.lib as L
from exp_robust.lib import OUT, NUM_CLASSES, CLASS_NAMES
from exp_robust.experiments import preattack_per_class
from exp_robust.blind_select import nearest_other_cluster

# ---------------------------------------------------------------- config
TAG = "q2"                  # namespace for this experiment's cache / attacks
MODEL_TAG = "baseline"
LR = 1e-4
COUNT2 = 50                 # per-non-victim cached samples (paper setting)

VICTIM_CLUSTER = 3          # majority class truck(9) in clusters/baseline
COUNTS = [2, 5, 10, 20]     # victim cached samples swept
EPOCHS = [5, 10, 20]        # attack epochs swept
REF_COUNT, REF_EPOCHS = 10, 20   # the paper's setting (timed precisely)


def count_trainable_server_params(server):
    """Replicate the attack's unfreeze rule (last 2 children of server.layer1)
    and count trainable vs total server params WITHOUT running the attack."""
    blocks = list(server.layer1.children())
    trainable = sum(p.numel() for blk in blocks[-2:] for p in blk.parameters())
    total = sum(p.numel() for p in server.parameters())
    return trainable, total


def time_normal_epoch():
    """Time + peak GPU memory of ONE normal USL training epoch on the real
    training loop (full 50k). Trains a throwaway 1-epoch model from scratch;
    per-epoch cost is convergence-independent. Deletes the probe model after."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    xtr, ytr, _, _ = L._base_cifar()
    t0 = time.time()
    try:
        L.train_split_model("overhead_probe", xtr, ytr, epochs=1)
    except Exception as e:                       # never let the probe block results
        print(f"[q2] normal-epoch probe failed: {e}", flush=True)
        shutil.rmtree(os.path.join(OUT, "models", "overhead_probe"), ignore_errors=True)
        return None
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6
    shutil.rmtree(os.path.join(OUT, "models", "overhead_probe"), ignore_errors=True)
    torch.cuda.empty_cache()
    return {"wall_s": float(wall), "peak_mem_mb": float(peak),
            "note": "1 epoch from scratch on full 50k train (per-epoch cost is "
                    "convergence-independent); the model LASER rides on is 120 of these."}


def main():
    model_dir = os.path.join(OUT, "models", MODEL_TAG)
    assert os.path.exists(os.path.join(model_dir, "server_nor_last.pt")), \
        "train baseline first"

    # ---- 0. parameter budget (no run) -------------------------------------
    net1, server, net3 = L.create_models()
    server.load_state_dict(torch.load(os.path.join(model_dir, "server_nor_last.pt")))
    srv_trainable, srv_total = count_trainable_server_params(server)
    whole = sum(p.numel() for m in (net1, server, net3) for p in m.parameters())
    del net1, server, net3
    torch.cuda.empty_cache()
    print(f"[q2] server f_s params: trainable(attack)={srv_trainable:,} / "
          f"{srv_total:,} = {100*srv_trainable/srv_total:.2f}%  "
          f"(whole model F={whole:,}; attack touches "
          f"{100*srv_trainable/whole:.2f}% of F)", flush=True)

    overhead = {
        "params": {
            "server_total": int(srv_total),
            "server_trainable_attack": int(srv_trainable),
            "server_trainable_frac": float(srv_trainable / srv_total),
            "whole_model_total": int(whole),
            "whole_model_trainable_frac": float(srv_trainable / whole),
        },
        "attack_data_budget": {
            "victim_samples": REF_COUNT,
            "nonvictim_per_class": COUNT2,
            "total_cached_vectors": REF_COUNT + COUNT2 * (NUM_CLASSES - 1),
            "fraction_of_train": float((REF_COUNT + COUNT2 * (NUM_CLASSES - 1)) / 50000),
            "client_bandwidth": "zero (reuses already-cached smashed data)",
        },
    }

    # ---- 1. build embedding cache ONCE (timed) ----------------------------
    cache_dir = os.path.join(OUT, "cache", TAG)
    xtr, ytr, _, _ = L._base_cifar()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    c2s_p, s2c_p, lab_p = L.collect_embeddings(TAG, xtr, ytr, model_dir)
    cache_wall = time.time() - t0
    cache_peak = torch.cuda.max_memory_allocated() / 1e6
    overhead["cache_build"] = {"wall_s": float(cache_wall),
                               "peak_mem_mb": float(cache_peak),
                               "note": "one-time no-grad forward over the 50k train "
                                       "set; reuses the forward pass the server runs "
                                       "anyway during normal training."}
    print(f"[q2] cache build: {cache_wall:.1f}s, peak {cache_peak:.0f} MB", flush=True)

    # ---- 2. reuse baseline partition + pick victim/proxy ------------------
    bcd = os.path.join(OUT, "clusters", "baseline")
    clabels = np.load(os.path.join(bcd, "cluster_labels.npy")).astype(np.int64)
    centers = np.load(os.path.join(bcd, "centers_pca.npy"))
    feat_path = os.path.join(bcd, "feat_pca.npy")
    info = json.load(open(os.path.join(bcd, "info.json")))
    c2c = info["cluster_to_class"]
    vc = int(c2c[str(VICTIM_CLUSTER)])
    pc = nearest_other_cluster(centers, VICTIM_CLUSTER,
                               forbidden_class=vc, cluster_to_class=c2c)
    pc_class = int(c2c[str(pc)])
    pre_per_class, base_acc = preattack_per_class(model_dir)
    print(f"[q2] victim cluster {VICTIM_CLUSTER} ({CLASS_NAMES[vc]}) -> proxy "
          f"cluster {pc} ({CLASS_NAMES[pc_class]}); base acc {base_acc:.2f}%",
          flush=True)

    # ---- 3. resource sweep (each attack timed) ----------------------------
    rows = []
    ref_attack = None
    for count in COUNTS:
        for ep in EPOCHS:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            t0 = time.time()
            save_dir = L.run_laser_attack(
                TAG, c2s_p, s2c_p, feat_path, centers, clabels,
                target_cluster=VICTIM_CLUSTER, proxy_cluster=pc,
                atk_epochs=ep, count=count, count2=COUNT2, lr=LR, model_tag=MODEL_TAG)
            atk_wall = time.time() - t0
            atk_peak = torch.cuda.max_memory_allocated() / 1e6

            post, acc_v, mp = L.evaluate_attack(TAG, model_dir, save_dir, ep, vc, pc_class)
            drop, pre_nv, post_nv = L.nva_drop(pre_per_class, post, vc)
            row = {
                "count": count, "epochs": ep,
                "acc_v": float(acc_v), "mp": float(mp), "nva_drop": float(drop),
                "attack_wall_s": float(atk_wall), "attack_peak_mem_mb": float(atk_peak),
            }
            rows.append(row)
            print(f"[q2] count={count:2d} ep={ep:2d}: ACC-V={acc_v:5.2f}%  "
                  f"MP={mp:5.2f}%  NVA={drop:5.2f}%  ({atk_wall:.1f}s, {atk_peak:.0f}MB)",
                  flush=True)
            if count == REF_COUNT and ep == REF_EPOCHS:
                ref_attack = row
            _save(rows, overhead, ref_attack)        # crash-safe incremental
            shutil.rmtree(save_dir, ignore_errors=True)
            torch.cuda.empty_cache()

    if ref_attack is not None:
        overhead["reference_attack"] = {k: ref_attack[k] for k in
                                        ("attack_wall_s", "attack_peak_mem_mb")}
        overhead["reference_attack"]["count"] = REF_COUNT
        overhead["reference_attack"]["epochs"] = REF_EPOCHS

    # ---- 4. normal-epoch comparison probe ---------------------------------
    overhead["normal_training_one_epoch"] = time_normal_epoch()

    # ---- 5. assemble ratios + save + free cache ---------------------------
    _finalize(rows, overhead, base_acc, info)
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"[q2] freed cache {cache_dir}", flush=True)
    _summarize(rows, overhead)


def _derived_ratios(overhead):
    ra = overhead.get("reference_attack", {})
    nt = overhead.get("normal_training_one_epoch") or {}
    out = {}
    if ra.get("attack_wall_s") and nt.get("wall_s"):
        out["attack_wall_over_one_train_epoch"] = ra["attack_wall_s"] / nt["wall_s"]
        out["est_full_train_wall_s"] = 120.0 * nt["wall_s"]
        out["attack_wall_over_full_train"] = ra["attack_wall_s"] / (120.0 * nt["wall_s"])
    return out


def _save(rows, overhead, ref_attack=None):
    out = {"tag": "q2_cost", "config": {
        "model": "VGG16", "split": "SB=1", "dataset": "CIFAR-10",
        "victim_cluster": VICTIM_CLUSTER, "victim_name": "truck",
        "counts": COUNTS, "epochs_grid": EPOCHS, "count2": COUNT2, "lr": LR,
    }, "overhead": overhead, "sweep_rows": rows}
    with open(os.path.join(OUT, "results_q2.json"), "w") as f:
        json.dump(out, f, indent=2)


def _finalize(rows, overhead, base_acc, info):
    overhead["ratios"] = _derived_ratios(overhead)
    out = {"tag": "q2_cost", "base_acc": float(base_acc),
           "clustering": {k: info[k] for k in ("k_star", "ari", "nmi", "purity")},
           "config": {
               "model": "VGG16", "split": "SB=1", "dataset": "CIFAR-10",
               "victim_cluster": VICTIM_CLUSTER, "victim_name": "truck",
               "counts": COUNTS, "epochs_grid": EPOCHS, "count2": COUNT2, "lr": LR,
           },
           "overhead": overhead, "sweep_rows": rows}
    with open(os.path.join(OUT, "results_q2.json"), "w") as f:
        json.dump(out, f, indent=2)
    _write_tables(rows, overhead, base_acc)


def _write_tables(rows, overhead, base_acc):
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)

    # --- overhead table ---
    p = overhead["params"]; db = overhead["attack_data_budget"]
    ra = overhead.get("reference_attack", {})
    cb = overhead.get("cache_build", {})
    nt = overhead.get("normal_training_one_epoch") or {}
    rt = overhead.get("ratios", {})
    path = os.path.join(OUT, "tables", "q2_overhead.md")
    with open(path, "w") as f:
        f.write("# LASER Attack Computational Overhead — VGG16, SB=1, CIFAR-10\n\n")
        f.write("The attack fine-tunes only the **last conv block** of the server, "
                "on a tiny subset of already-cached smashed data. The headline "
                "cost metric is the **trainable-parameter fraction** (architecture-"
                "portable); absolute timings are for the VGG16/SB=1 reference.\n\n")
        f.write("## Parameter budget\n\n")
        f.write("| Quantity | Value |\n|---|---|\n")
        f.write("| Server f_s total params | %s |\n" % f"{p['server_total']:,}")
        f.write("| Trainable during attack (last conv block) | %s (%.2f%% of f_s) |\n"
                % (f"{p['server_trainable_attack']:,}", 100 * p["server_trainable_frac"]))
        f.write("| Whole model F params | %s |\n" % f"{p['whole_model_total']:,}")
        f.write("| Attack touches | %.2f%% of the whole model |\n"
                % (100 * p["whole_model_trainable_frac"]))
        f.write("\n## Time / memory (reference attack: %d victim samples, %d epochs)\n\n"
                % (REF_COUNT, REF_EPOCHS))
        f.write("| Component | Wall-clock | Peak GPU mem |\n|---|---|---|\n")
        f.write("| Embedding cache build (one-time, 50k fwd) | %.1f s | %.0f MB |\n"
                % (cb.get("wall_s", float("nan")), cb.get("peak_mem_mb", float("nan"))))
        f.write("| **LASER fine-tune (20 ep, %d vectors)** | **%.1f s** | **%.0f MB** |\n"
                % (db["total_cached_vectors"], ra.get("attack_wall_s", float("nan")),
                   ra.get("attack_peak_mem_mb", float("nan"))))
        if nt:
            f.write("| Normal training, 1 epoch (full 50k) | %.1f s | %.0f MB |\n"
                    % (nt["wall_s"], nt["peak_mem_mb"]))
            f.write("| Normal training, full 120 ep (est.) | %.0f s | — |\n"
                    % (120.0 * nt["wall_s"]))
        f.write("\n## Data budget\n\n")
        f.write("Attack uses **%d cached vectors** (%d victim + %d×%d non-victim) "
                "= %.2f%% of the 50k train set, with **zero client bandwidth** "
                "(reuses smashed data the server already computed).\n\n"
                % (db["total_cached_vectors"], db["victim_samples"], NUM_CLASSES - 1,
                   db["nonvictim_per_class"], 100 * db["fraction_of_train"]))
        if rt:
            f.write("**Ratios:** attack wall-clock = %.1f%% of one training epoch; "
                    "= %.3f%% of full 120-epoch training.\n"
                    % (100 * rt["attack_wall_over_one_train_epoch"],
                       100 * rt["attack_wall_over_full_train"]))
    print(f"[q2] overhead table -> {path}", flush=True)

    # --- resource sweep grid (ACC-V) ---
    path = os.path.join(OUT, "tables", "q2_sweep.md")
    with open(path, "w") as f:
        f.write("# LASER Resource-Constraint Sweep — VGG16, SB=1, CIFAR-10\n\n")
        f.write("Victim = truck cluster (blind, nearest-centroid proxy). Base acc "
                "**%.2f%%**. Each cell = post-attack **Victim Accuracy (ACC-V)**; "
                "lower = more destroyed (paper target = 0%%).\n\n" % base_acc)
        f.write("| victim samples \\ epochs | " +
                " | ".join("%d" % e for e in EPOCHS) + " |\n")
        f.write("|" + "---|" * (len(EPOCHS) + 1) + "\n")
        for c in COUNTS:
            cells = []
            for e in EPOCHS:
                r = next(r for r in rows if r["count"] == c and r["epochs"] == e)
                mark = " **(paper)**" if (c == REF_COUNT and e == REF_EPOCHS) else ""
                cells.append("%.2f%%%s" % (r["acc_v"], mark))
            f.write("| %d | %s |\n" % (c, " | ".join(cells)))
        f.write("\nMisclassification (MP) and collateral (NVA-Drop) for every cell "
                "are in `results_q2.json`.\n\n")
        low = min(rows, key=lambda r: (r["count"], r["epochs"]))
        destroyed = sum(1 for r in rows if r["acc_v"] <= 1.0)
        f.write("**Summary:** %d/%d cells destroy the victim (ACC-V ≤ 1%%). Even the "
                "most constrained cell (%d samples, %d epochs) reaches ACC-V = "
                "%.2f%% (MP %.1f%%, NVA %.2f%%) — the attack succeeds under tight "
                "resource budgets.\n"
                % (destroyed, len(rows), low["count"], low["epochs"], low["acc_v"],
                   low["mp"], low["nva_drop"]))
    print(f"[q2] sweep table -> {path}", flush=True)


def _summarize(rows, overhead):
    p = overhead["params"]; ra = overhead.get("reference_attack", {})
    nt = overhead.get("normal_training_one_epoch") or {}
    destroyed = sum(1 for r in rows if r["acc_v"] <= 1.0)
    print("\n[q2] DONE.", flush=True)
    print(f"  params: attack trains {100*p['server_trainable_frac']:.2f}% of f_s "
          f"({100*p['whole_model_trainable_frac']:.2f}% of whole model)", flush=True)
    if ra and nt:
        print(f"  attack {ra.get('attack_wall_s',0):.1f}s vs 1 train epoch "
              f"{nt.get('wall_s',0):.1f}s "
              f"({100*ra.get('attack_wall_s',0)/max(nt.get('wall_s',1),1e-9):.1f}%)",
              flush=True)
    print(f"  sweep: {destroyed}/{len(rows)} cells destroy victim (ACC-V<=1%)",
          flush=True)


if __name__ == "__main__":
    main()
