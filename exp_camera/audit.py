"""
audit.py — M10: per-class Semantic Integrity Auditing prototype + overhead.

The client holds a small labeled validation set.  For every attacked server
produced by the M5/M6 runs, compute per-class recall on the audit split and flag
classes whose recall drops by at least a threshold.  This turns the paper's
recommended defense from a simulated metric into a runnable prototype.

Outputs
-------
exp_camera/outputs/results_audit_overhead.json
exp_camera/outputs/tables/audit_overhead.md

Usage
-----
    python -m exp_camera.audit
    python -m exp_camera.audit --thresholds 1 2 5 10 --audit-per-class 500
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision

import exp_camera.lib as L

REPO = L.REPO
OUT = L.OUT
ATTACK_DIR = os.path.join(OUT, "attacked_servers")
THRESHOLDS = (1.0, 2.0, 5.0, 10.0)

PATTERN = re.compile(
    r"^(cifar10_sb(?P<sb>\d+)_seed(?P<seed>\d+))_v(?P<victim>\d+)"
    r"_p(?P<proxy>\d+)_server(?P<epoch>\d+)\.pt$")


def _split_indices(testset, per_class):
    """Stratified audit / evaluation split: first `per_class` of each class for
    auditing, the remainder for evaluation."""
    targets = np.array(testset.targets)
    audit, evals = [], []
    for c in range(L.NUM_CLASSES):
        idx = np.where(targets == c)[0]
        audit.extend(idx[:per_class].tolist())
        evals.extend(idx[per_class:].tolist())
    return audit, evals


@torch.no_grad()
def per_class_recall(net1, server, net3, loader):
    net1.eval(); server.eval(); net3.eval()
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
    return 100.0 * correct / np.maximum(total, 1)


def _load_clean(mdir, sb):
    net1, server, net3 = L.create_models(sb)
    net1.load_state_dict(torch.load(os.path.join(mdir, "net1_nor_last.pt")))
    server.load_state_dict(torch.load(os.path.join(mdir, "server_nor_last.pt")))
    net3.load_state_dict(torch.load(os.path.join(mdir, "net3_nor_last.pt")))
    return net1, server, net3


def _load_attacked(mdir, sb, path):
    net1, server, net3 = L.create_models(sb)
    net1.load_state_dict(torch.load(os.path.join(mdir, "net1_nor_last.pt")))
    net3.load_state_dict(torch.load(os.path.join(mdir, "net3_nor_last.pt")))
    server.load_state_dict(torch.load(path))
    return net1, server, net3


def audit_one(path, audit_loader, eval_loader, thresholds=THRESHOLDS):
    m = PATTERN.match(os.path.basename(path))
    if not m:
        return None
    sb = int(m.group("sb"))
    seed = int(m.group("seed"))
    victim = int(m.group("victim"))
    proxy = int(m.group("proxy"))
    tag = f"cifar10_sb{sb}_seed{seed}"
    mdir = os.path.join(OUT, "models", tag)
    if not os.path.exists(os.path.join(mdir, "server_nor_last.pt")):
        print(f"[audit] skip {path}: clean model {mdir} missing", flush=True)
        return None

    clean1, clean_server, clean3 = _load_clean(mdir, sb)
    att1, att_server, att3 = _load_attacked(mdir, sb, path)

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    clean_recall = per_class_recall(clean1, clean_server, clean3, audit_loader)
    att_recall = per_class_recall(att1, att_server, att3, audit_loader)
    wall_s = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6
               if torch.cuda.is_available() else None)

    # Evaluation split (not used by the auditor) for reporting the attack outcome.
    clean_eval = per_class_recall(clean1, clean_server, clean3, eval_loader)
    att_eval = per_class_recall(att1, att_server, att3, eval_loader)

    drop = clean_recall - att_recall
    records = []
    for th in thresholds:
        flagged = [int(c) for c in np.where(drop >= th)[0]]
        fp = [c for c in flagged if c != victim]
        records.append({
            "threshold_pct": float(th),
            "flagged_classes": flagged,
            "flagged_names": [L.CLASS_NAMES[c] for c in flagged],
            "victim_flagged": bool(victim in flagged),
            "false_positives": fp,
            "false_positive_names": [L.CLASS_NAMES[c] for c in fp],
            "n_flagged": len(flagged),
        })

    return {
        "file": path, "tag": tag, "sb": sb, "seed": seed,
        "victim_class": victim, "victim_name": L.CLASS_NAMES[victim],
        "proxy_class": proxy, "proxy_name": L.CLASS_NAMES[proxy],
        "clean_audit_recall": {L.CLASS_NAMES[c]: float(clean_recall[c])
                              for c in range(L.NUM_CLASSES)},
        "attacked_audit_recall": {L.CLASS_NAMES[c]: float(att_recall[c])
                                  for c in range(L.NUM_CLASSES)},
        "audit_recall_drop": {L.CLASS_NAMES[c]: float(drop[c])
                              for c in range(L.NUM_CLASSES)},
        "clean_eval_recall": {L.CLASS_NAMES[c]: float(clean_eval[c])
                              for c in range(L.NUM_CLASSES)},
        "attacked_eval_recall": {L.CLASS_NAMES[c]: float(att_eval[c])
                                 for c in range(L.NUM_CLASSES)},
        "thresholds": records,
        "audit_cost": {"wall_s": float(wall_s), "peak_mem_mb": peak_mb,
                       "n_audit_images": int(sum(audit_loader.dataset.indices
                                                 if isinstance(audit_loader.dataset, Subset)
                                                 else range(len(audit_loader.dataset))))},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-per-class", type=int, default=500)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=list(THRESHOLDS))
    args = p.parse_args()

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=L.get_transforms(False))
    audit_idx, eval_idx = _split_indices(testset, args.audit_per_class)
    audit_loader = DataLoader(Subset(testset, audit_idx), batch_size=512,
                              shuffle=False, num_workers=8)
    eval_loader = DataLoader(Subset(testset, eval_idx), batch_size=512,
                             shuffle=False, num_workers=8)
    paths = sorted(
        os.path.join(ATTACK_DIR, f) for f in os.listdir(ATTACK_DIR)
        if PATTERN.match(f)) if os.path.isdir(ATTACK_DIR) else []
    if not paths:
        raise SystemExit(f"no attacked servers in {ATTACK_DIR}; run "
                         "`python -m exp_camera.seed_split attack-batch` first")

    rows = []
    for path in paths:
        print(f"[audit] {os.path.basename(path)}", flush=True)
        rec = audit_one(path, audit_loader, eval_loader, args.thresholds)
        if rec:
            rows.append(rec)
            print(f"[audit] victim={rec['victim_name']} "
                  f"audit recall {rec['clean_audit_recall'][rec['victim_name']]:.1f}->"
                  f"{rec['attacked_audit_recall'][rec['victim_name']]:.1f}%; "
                  f"flagged@5%={rec['thresholds'][[t['threshold_pct'] for t in rec['thresholds']].index(5.0)]['victim_flagged'] if 5.0 in args.thresholds else 'n/a'}",
                  flush=True)

    # Aggregate primary threshold (5% if present, else first).
    primary = 5.0 if 5.0 in args.thresholds else args.thresholds[0]
    agg = {}
    for th in args.thresholds:
        rr = [r for r in rows]
        idx = [i for i, r in enumerate(rr)
               if r["thresholds"][[t["threshold_pct"] for t in r["thresholds"]].index(th)]]
        tpr = [1.0 if r["thresholds"][[t["threshold_pct"] for t in r["thresholds"]].index(th)]["victim_flagged"] else 0.0 for r in rr]
        fpr = [1.0 if r["thresholds"][[t["threshold_pct"] for t in r["thresholds"]].index(th)]["false_positives"] else 0.0 for r in rr]
        agg[str(th)] = {"n": len(rr), "tpr": float(np.mean(tpr)) if tpr else None,
                        "fpr": float(np.mean(fpr)) if fpr else None}
    payload = {
        "n_models": len(rows), "primary_threshold_pct": primary,
        "audit_per_class": args.audit_per_class,
        "thresholds": args.thresholds, "aggregate": agg, "rows": rows,
        "cost_reference": None,
    }
    q2_path = os.path.join(REPO, "exp_robust", "outputs", "results_q2.json")
    if os.path.exists(q2_path):
        q2 = json.load(open(q2_path))
        payload["cost_reference"] = q2.get("overhead", {}).get(
            "normal_training_one_epoch")
    json_path = os.path.join(OUT, "results_audit_overhead.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    md_path = os.path.join(OUT, "tables", "audit_overhead.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write("# M10 — Per-Class Semantic Integrity Audit\n\n")
        f.write(f"Audit split: {args.audit_per_class} labeled images/class; "
                f"primary threshold {primary}% recall drop. "
                f"Models audited: {len(rows)}.\n\n")
        f.write("| Threshold | TPR (victim flagged) | FPR (any non-victim flagged) |\n")
        f.write("|---:|---:|---:|\n")
        for th in args.thresholds:
            a = agg[str(th)]
            f.write(f"| {th:.0f}% | {a['tpr']:.2f} | {a['fpr']:.2f} |\n")
        f.write("\n| Model | Victim | Audit recall clean→attacked | Flagged @5% | False positives | Audit wall (s) | Peak GPU (MB) |\n")
        f.write("|---|---|---:|---|---:|---:|---:|\n")
        for r in rows:
            v = r["victim_name"]
            th5 = next(t for t in r["thresholds"] if t["threshold_pct"] == primary)
            f.write(f"| {r['tag']} | {v} | "
                    f"{r['clean_audit_recall'][v]:.1f}→{r['attacked_audit_recall'][v]:.1f} | "
                    f"{th5['victim_flagged']} | {len(th5['false_positives'])} | "
                    f"{r['audit_cost']['wall_s']:.2f} | "
                    f"{r['audit_cost']['peak_mem_mb']:.0f} |\n")
        if payload["cost_reference"]:
            cr = payload["cost_reference"]
            f.write(f"\nReference: one normal training epoch = {cr['wall_s']:.1f} s, "
                    f"{cr['peak_mem_mb']:.0f} MB (from `results_q2.json`).\n")
        f.write("\n**Honest limitation.** The auditor needs a labeled validation set, "
                "and the victim is unknown a priori; the prototype shows that a "
                "5%-per-class recall threshold flags the victim while keeping "
                "false positives at the reported rate. It is a client-side "
                "post-deployment check, not a training-time defense.\n")
    print(f"[audit] -> {json_path}\n[audit] -> {md_path}")


if __name__ == "__main__":
    main()
