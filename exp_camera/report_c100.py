"""
report_c100.py — M7 combined CIFAR-100 split-depth table (SB=1/2/3).

SB=1 is the existing exp_scale result; SB=2/3 are the new M7 runs.  Produces:

    exp_camera/outputs/tables/split_cifar100.md
    exp_camera/outputs/summary_split_cifar100.json
"""
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "exp_camera", "outputs")


def _summ(rows, clustering, base_acc, sb, source):
    recalls = [r["victim_recall_post"] for r in rows]
    mp = [r["mp"] for r in rows]
    nva = [r["nva_drop"] for r in rows]
    return {
        "sb": sb, "source": source, "base_acc": base_acc,
        "k_star": clustering["k_star"], "ari": clustering["ari"],
        "nmi": clustering["nmi"], "purity": clustering["purity"],
        "n": len(rows),
        "le1": int(sum(x <= 1.0 for x in recalls)),
        "le5": int(sum(x <= 5.0 for x in recalls)),
        "mean_mp": float(np.mean(mp)), "mean_nva": float(np.mean(nva)),
        "max_nva": float(np.max(nva)),
        "rows": rows,
    }


def main():
    runs = []
    p1 = os.path.join(REPO, "exp_scale", "outputs",
                      "results_scale_cifar100_20ep.json")
    d1 = json.load(open(p1))
    runs.append(_summ(d1["rows"], d1["clustering"], d1["base_acc"], 1,
                      "exp_scale SB=1 (existing)"))
    for sb in (2, 3):
        p = os.path.join(OUT, "cifar100",
                         f"results_cifar100_sb{sb}_seed9202.json")
        d = json.load(open(p))
        runs.append(_summ(d["rows"], d["clustering"], d["base_acc"], sb,
                          "exp_camera M7 (new)"))

    md = os.path.join(OUT, "tables", "split_cifar100.md")
    os.makedirs(os.path.dirname(md), exist_ok=True)
    with open(md, "w") as f:
        f.write("# M7 — CIFAR-100 split depth (VGG-16, seed 9202)\n\n")
        f.write("Blind label-free victim selection: 12 clusters by size-rank "
                "stratification; proxy = nearest PCA centroid; garbage blobs "
                "excluded label-free. K-search ceiling max_k=120. Attack: 10 "
                "victim + 50/non-victim cached vectors, 20 epochs, λ=1, α=0.5. "
                "Ground-truth labels are evaluation-only. SB=1 is the existing "
                "exp_scale result; SB=2/3 are new single-seed runs.  Base Acc is "
                "the clean `nor_last` accuracy used for attack evaluation.\n\n")
        f.write("| Cut | SB | n | Base Acc | K* | ARI | NMI | Purity | "
                "Recall ≤5% | Recall ≤1% | Mean MP | Mean NVA | Max NVA |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        names = {1: "Shallow", 2: "Middle", 3: "Deep"}
        for r in runs:
            f.write(f"| {names[r['sb']]} | {r['sb']} | {r['n']} | "
                    f"{r['base_acc']:.2f} | {r['k_star']} | {r['ari']:.3f} | "
                    f"{r['nmi']:.3f} | {r['purity']:.3f} | "
                    f"{r['le5']}/{r['n']} | {r['le1']}/{r['n']} | "
                    f"{r['mean_mp']:.2f}% | {r['mean_nva']:.2f}% | "
                    f"{r['max_nva']:.2f}% |\n")
        f.write("\n### Per-cluster recalls\n\n")
        f.write("| SB | Victim | Purity | Size | Recall pre→post | MP | NVA |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for r in runs:
            for row in sorted(r["rows"], key=lambda x: -x["victim_recall_post"]):
                f.write(f"| {r['sb']} | {row['victim_name']} | "
                        f"{row['cluster_purity']:.2f} | {row['cluster_size']} | "
                        f"{row['victim_recall_pre']:.1f} → "
                        f"**{row['victim_recall_post']:.2f}** | "
                        f"{row['mp']:.2f}% | {row['nva_drop']:.2f}% |\n")
        f.write("\n**Reading.** Deep splits do not break the attack: 10–11/12 "
                "clusters reach ≤5% recall and mean NVA stays 0.59–1.00%. "
                "Clustering quality degrades modestly with deeper cuts "
                "(K* 102→91→95, ARI 0.82→0.78/0.80, purity 0.91→0.85/0.88); "
                "the residual is cluster-specific (flatfish; whale/rabbit), not "
                "a global collapse.\n")
    with open(os.path.join(OUT, "summary_split_cifar100.json"), "w") as f:
        json.dump({"runs": runs}, f, indent=2)
    print(f"[M7-report] -> {md}")


if __name__ == "__main__":
    main()
