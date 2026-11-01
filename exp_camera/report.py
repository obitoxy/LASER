"""
report.py — M5/M6 tables from the controlled CIFAR-10 results.

Reads exp_camera/outputs/results_cifar10_sb{sb}_seed{seed}.json and writes:

    exp_camera/outputs/tables/seed_vgg16.md
    exp_camera/outputs/tables/split_vgg16.md
    exp_camera/outputs/summary_seed_split.json

Mean ± sample std (ddof=1); 95% CI uses the Student-t critical value for n=3
(4.303).  With n=1 the CI is reported as n/a.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import t as t_dist

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "exp_camera", "outputs")
SEEDS = (9202, 9203, 9204)
SPLITS = (1, 2, 3)
METRICS = ["base_acc", "acc_v", "mp", "nva_drop", "global_drop",
           "k_star", "ari", "nmi", "purity", "cluster_purity", "cluster_size"]


def load_results():
    out = []
    for path in sorted(glob.glob(os.path.join(
            OUT, "results_cifar10_sb*_seed*.json"))):
        d = json.load(open(path))
        if d.get("status") != "done" or "attack" not in d:
            continue
        a = d["attack"]
        c = d.get("clustering", {})
        out.append({
            "tag": d["tag"], "sb": int(d["split_block"]), "seed": int(d["seed"]),
            "base_acc": d.get("base_acc"),
            "acc_v": a.get("acc_v"), "mp": a.get("mp"),
            "nva_drop": a.get("nva_drop"),
            "global_drop": (None if d.get("base_acc") is None
                            or a.get("global_acc_post") is None
                            else d["base_acc"] - a["global_acc_post"]),
            "k_star": c.get("k_star"), "ari": c.get("ari"),
            "nmi": c.get("nmi"), "purity": c.get("purity"),
            "cluster_purity": a.get("cluster_purity"),
            "cluster_size": a.get("cluster_size"),
            "victim": a.get("victim_name"), "proxy": a.get("proxy_name"),
        })
    return out


def stats(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    mean = float(np.mean(vals))
    if n == 1:
        return {"n": 1, "mean": mean, "std": None, "ci95": None}
    std = float(np.std(vals, ddof=1))
    half = float(t_dist.ppf(0.975, n - 1) * std / np.sqrt(n))
    return {"n": n, "mean": mean, "std": std,
            "ci95": [mean - half, mean + half]}


def fmt_mean(s, digits=2):
    if s["n"] == 0:
        return "—"
    if s["n"] == 1:
        return f"{s['mean']:.{digits}f} (n=1)"
    return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"


def fmt_ci(s, digits=2):
    if s["ci95"] is None:
        return "—"
    return f"[{s['ci95'][0]:.{digits}f}, {s['ci95'][1]:.{digits}f}]"


def write_seed(rows, path):
    by_sb1 = sorted([r for r in rows if r["sb"] == 1], key=lambda r: r["seed"])
    with open(path, "w") as f:
        f.write("# M5 — Seed variance (VGG-16 / CIFAR-10 / SB=1)\n\n")
        f.write("Controlled protocol: 120 epochs (SGD+mixup); attack: LASER "
                "last-conv-block, guided centroid λ=1, anchor α=0.5, 20 epochs, "
                "10 victim + 50/non-victim cached vectors, fixed victim/proxy "
                "truck→car. Ground-truth labels are evaluation-only. The paper's "
                "main table uses a 400-epoch protocol, so this table is a "
                "controlled seed study, not a re-run of that row.\n\n")
        f.write("| Seed | Base Acc | ACC-V | MP | NVA Drop | Global Drop | K* | ARI | NMI | Purity | Victim cluster purity | Size |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in by_sb1:
            f.write(f"| {r['seed']} | {r['base_acc']:.2f} | {r['acc_v']:.2f} | "
                    f"{r['mp']:.2f} | {r['nva_drop']:.2f} | "
                    f"{r['global_drop']:.2f} | {r['k_star']} | {r['ari']:.3f} | "
                    f"{r['nmi']:.3f} | {r['purity']:.3f} | "
                    f"{r['cluster_purity']:.3f} | {r['cluster_size']} |\n")
        f.write("\n| Aggregate | Value |\n|---|---:|\n")
        for m in METRICS:
            s = stats([r[m] for r in by_sb1])
            f.write(f"| {m} (mean ± std) | {fmt_mean(s, 3 if m in ('ari','nmi','purity','cluster_purity') else 2)} |\n")
        f.write("\n| Metric | 95% CI (t, n−1) |\n|---|---:|\n")
        for m in ["base_acc", "acc_v", "mp", "nva_drop", "ari", "purity"]:
            s = stats([r[m] for r in by_sb1])
            f.write(f"| {m} | {fmt_ci(s)} |\n")
        f.write(f"\n**n = {len(by_sb1)} seeds.** Seed list: "
                f"{', '.join(str(r['seed']) for r in by_sb1)}.\n")


def write_split(rows, path):
    with open(path, "w") as f:
        f.write("# M6 — Split depth (VGG-16 / CIFAR-10, 3 seeds per cut)\n\n")
        f.write("Same controlled protocol as M5. Each cell is mean ± std over the "
                "three seeds. Fixed victim/proxy truck→car; ground-truth labels "
                "are evaluation-only. Base Acc is measured on the clean `nor_last` "
                "model used for the attack; the paper's 400-epoch main-table row is "
                "not overwritten.\n\n")
        f.write("| Cut | SB | n | Base Acc | K* | ARI | NMI | Purity | ACC-V | MP | NVA Drop | Victim cluster purity | Size |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        names = {1: "Shallow", 2: "Middle", 3: "Deep"}
        for sb in SPLITS:
            sub = [r for r in rows if r["sb"] == sb]
            vals = {m: stats([r[m] for r in sub]) for m in METRICS}
            f.write(f"| {names[sb]} | {sb} | {len(sub)} | "
                    f"{fmt_mean(vals['base_acc'])} | {fmt_mean(vals['k_star'], 1)} | "
                    f"{fmt_mean(vals['ari'], 3)} | {fmt_mean(vals['nmi'], 3)} | "
                    f"{fmt_mean(vals['purity'], 3)} | {fmt_mean(vals['acc_v'])} | "
                    f"{fmt_mean(vals['mp'])} | {fmt_mean(vals['nva_drop'])} | "
                    f"{fmt_mean(vals['cluster_purity'], 3)} | "
                    f"{fmt_mean(vals['cluster_size'], 0)} |\n")
        f.write("\n### Per-run values\n\n")
        f.write("| SB | Seed | Base Acc | K* | ARI | Purity | ACC-V | MP | NVA Drop |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted(rows, key=lambda r: (r["sb"], r["seed"])):
            f.write(f"| {r['sb']} | {r['seed']} | {r['base_acc']:.2f} | "
                    f"{r['k_star']} | {r['ari']:.3f} | {r['purity']:.3f} | "
                    f"{r['acc_v']:.2f} | {r['mp']:.2f} | {r['nva_drop']:.2f} |\n")


def main():
    rows = load_results()
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    seed_path = os.path.join(OUT, "tables", "seed_vgg16.md")
    split_path = os.path.join(OUT, "tables", "split_vgg16.md")
    write_seed(rows, seed_path)
    write_split(rows, split_path)
    summary = {"rows": rows,
               "by_sb": {str(sb): {m: stats([r[m] for r in rows if r["sb"] == sb])
                                   for m in METRICS} for sb in SPLITS},
               "by_seed_sb1": {str(seed): {m: stats([r[m] for r in rows
                                                       if r["sb"] == 1 and r["seed"] == seed])
                                           for m in METRICS} for seed in SEEDS}}
    with open(os.path.join(OUT, "summary_seed_split.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[report] {len(rows)} runs -> {seed_path}\n[report] -> {split_path}")


if __name__ == "__main__":
    main()
