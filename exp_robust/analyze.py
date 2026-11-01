"""
analyze.py — Aggregate the per-condition JSON results into tables and figures
for the RAID 2026 rebuttal, then render a few key plots.

Reads exp_robust/outputs/results_*.json and writes:
    exp_robust/outputs/tables/*.txt   (copy-pasteable LaTeX-ish tables)
    exp_robust/outputs/figs/*.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
TBL = os.path.join(OUT, "tables"); os.makedirs(TBL, exist_ok=True)
FIG = os.path.join(OUT, "figs"); os.makedirs(FIG, exist_ok=True)

CN = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

C_MAIN = "#1f5fa8"      # clean/baseline blue
C_IMB  = "#c0504d"      # imbalance red
C_NOISE= "#7f7f2d"      # noise olive
C_DYN  = "#4c9f70"      # dynamic green


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def collect_attack_rows(res):
    """Flatten attacks into per-victim rows."""
    rows = []
    for a in res.get("attacks", []):
        st = a.get("status", "ok")
        if st == "NO_DEDICATED_CLUSTER":
            rows.append({"victim": a["victim_name"], "victim_cls": a["victim_class"],
                         "acc_v": None, "mp": None, "nva_drop": None,
                         "status": "no-cluster"})
        elif st == "IMPURE_CLUSTER":
            rows.append({"victim": a["victim_name"], "victim_cls": a["victim_class"],
                         "acc_v": None, "mp": None, "nva_drop": None,
                         "status": "impure", "purity": a.get("cluster_purity", 0.0)})
        else:
            rows.append({"victim": a["victim_name"], "victim_cls": a["victim_class"],
                         "proxy": a["proxy_name"], "acc_v": a["acc_v"], "mp": a["mp"],
                         "nva_drop": a["nva_drop"], "status": "ok"})
    return rows


def make_overview_table():
    """Big table: per condition, clustering quality + per-victim attack metrics."""
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 1: LASER under complex / non-ideal data distributions (VGG16, SB=1, CIFAR-10)")
    lines.append("=" * 100)
    header = f"{'Condition':<16}{'BaseAcc':>8}{'K*':>5}{'ARI':>7}{'NMI':>7}{'Purity':>8}   {'Victim':<9}{'ACC-V':>8}{'MP':>8}{'NVA-Drop':>10}  note"
    lines.append(header)
    lines.append("-" * 100)

    files = [("Clean (control)", "results_baseline.json"),
             ("Imbalance ρ=10", "results_lt10.json"),
             ("Imbalance ρ=100", "results_lt.json"),
             ("Noise 20%", "results_noise20.json"),
             ("Noise 40%", "results_noise40.json")]
    for label, fn in files:
        res = load(fn)
        if res is None:
            lines.append(f"{label:<16}  (not run yet)")
            continue
        cl = res["clustering"]
        rows = collect_attack_rows(res)
        for i, r in enumerate(rows):
            cond = label if i == 0 else ""
            base = f"{res.get('base_acc',0):.2f}%" if i == 0 else ""
            ks = f"{cl['k_star']}" if i == 0 else ""
            ari = f"{cl['ari']:.3f}" if i == 0 else ""
            nmi = f"{cl['nmi']:.3f}" if i == 0 else ""
            pur = f"{cl['purity']:.3f}" if i == 0 else ""
            if r["status"] == "no-cluster":
                lines.append(f"{cond:<16}{base:>8}{ks:>5}{ari:>7}{nmi:>7}{pur:>8}   "
                             f"{r['victim']:<9}{'--':>8}{'--':>8}{'--':>10}  no dedicated cluster")
            elif r["status"] == "impure":
                lines.append(f"{cond:<16}{base:>8}{ks:>5}{ari:>7}{nmi:>7}{pur:>8}   "
                             f"{r['victim']:<9}{'--':>8}{'--':>8}{'--':>10}  impure cluster "
                             f"(p={r['purity']:.2f})")
            else:
                lines.append(f"{cond:<16}{base:>8}{ks:>5}{ari:>7}{nmi:>7}{pur:>8}   "
                             f"{r['victim']:<9}{r['acc_v']:>7.2f}%{r['mp']:>7.1f}%"
                             f"{r['nva_drop']:>9.2f}%  ->{r['proxy']}")
    lines.append("=" * 100)
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(TBL, "table1_overview.txt"), "w") as f:
        f.write(txt + "\n")


def make_dynamic_table():
    res = load("results_dynamic.json")
    if res is None:
        print("dynamic not run yet"); return
    lines = ["=" * 95,
             "TABLE 2: Dynamic distribution — 3-phase non-stationary client stream",
             "=" * 95]
    lines.append(f"{'Phase':<24}{'#active':>8}{'K*':>5}{'ARI':>7}{'Purity':>8}"
                 f"   {'Victim':<8}{'ACC-V':>8}{'MP':>8}{'NVA-Drop':>10}")
    lines.append("-" * 95)
    for p in res["phases"]:
        if p.get("status") == "NO_DEDICATED_CLUSTER":
            lines.append(f"{p['phase']:<24}{p.get('true_n_active',''):>8}{p['k_star']:>5}"
                         f"{p['ari']:>7.3f}{p['purity']:>8.3f}   no dedicated cluster")
        else:
            lines.append(f"{p['phase']:<24}{p['true_n_active']:>8}{p['k_star']:>5}"
                         f"{p['ari']:>7.3f}{p['purity']:>8.3f}   "
                         f"{p['victim_name']:<8}{p['acc_v']:>7.2f}%{p['mp']:>7.1f}%"
                         f"{p['nva_drop']:>9.2f}%")
    lines.append("=" * 95)
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(TBL, "table2_dynamic.txt"), "w") as f:
        f.write(txt + "\n")


def plot_silhouette_curves():
    """Silhouette(K) curve per condition — shows K-search robustness."""
    plt.figure(figsize=(7, 4.5))
    files = [("Clean", "results_baseline.json", C_MAIN),
             ("Imbalance ρ=100", "results_lt.json", C_IMB),
             ("Noise 20%", "results_noise20.json", C_NOISE),
             ("Noise 40%", "results_noise40.json", "#b276b2")]
    for label, fn, col in files:
        res = load(fn)
        if res is None:
            continue
        curve = res["clustering"].get("sil_curve", {})
        if not curve:
            continue
        ks = sorted(int(k) for k in curve)
        ys = [curve[str(k)] if str(k) in curve else curve[k] for k in ks]
        plt.plot(ks, ys, "-o", color=col, label=f"{label} (k*={res['clustering']['k_star']})")
    plt.xlabel("Number of clusters K"); plt.ylabel("Silhouette score")
    plt.title("Silhouette-based K search across data distributions")
    plt.xticks(range(2, 13)); plt.grid(alpha=0.3); plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "silhouette_curves.png"), dpi=150)
    plt.close()
    print("saved silhouette_curves.png")


def plot_attack_bar():
    """Grouped bar: ACC-V (lower=better) and NVA-Drop (lower=better) per condition."""
    conds = []
    accv = []; nvad = []; labels = []
    for label, fn in [("Clean", "results_baseline.json"),
                      ("Imb. ρ=100", "results_lt.json"),
                      ("Noise 20%", "results_noise20.json"),
                      ("Noise 40%", "results_noise40.json")]:
        res = load(fn)
        if res is None:
            continue
        for r in collect_attack_rows(res):
            if r["status"] == "ok":
                labels.append(f"{label}\n{r['victim']}")
                accv.append(r["acc_v"])
                nvad.append(r["nva_drop"])
    if not labels:
        return
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 4.5))
    ax.bar(x - w/2, accv, w, label="ACC-V (victim, ↓ better)", color="#c0504d")
    ax.bar(x + w/2, nvad, w, label="NVA-Drop (collateral, ↓ better)", color="#1f5fa8")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%) / Drop (%)")
    ax.set_title("LASER effectiveness vs. stealth across distributions")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "attack_bar.png"), dpi=150)
    plt.close()
    print("saved attack_bar.png")


def plot_dynamic():
    res = load("results_dynamic.json")
    if res is None:
        return
    phases = [p["phase"].replace("phase","P").replace("_"," ") for p in res["phases"]]
    kstar = [p["k_star"] for p in res["phases"]]
    nact = [p["true_n_active"] for p in res["phases"]]
    accv = [p.get("acc_v", 0) for p in res["phases"]]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(phases))
    ax1.bar(x - 0.2, nact, 0.4, label="#active classes (true)", color="#9ecae1")
    ax1.bar(x + 0.2, kstar, 0.4, label="K* (silhouette, unsupervised)", color="#1f5fa8")
    ax1.set_xticks(x); ax1.set_xticklabels(phases, fontsize=9)
    ax1.set_ylabel("number of clusters"); ax1.legend(loc="upper left", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(x, accv, "-o", color="#c0504d", label="Victim ACC-V after LASER")
    ax2.set_ylabel("ACC-V (%)", color="#c0504d")
    ax2.set_ylim(-5, 105)
    ax2.legend(loc="upper right", fontsize=9)
    plt.title("Dynamic stream: LASER K-search tracks the active classes & stays effective")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "dynamic.png"), dpi=150)
    plt.close()
    print("saved dynamic.png")


def plot_lt_per_class():
    res = load("results_lt.json")
    if res is None:
        return
    # per-cluster purity vs cluster size -> minority classes have small/impure clusters
    per = res["clustering"]["per_cluster"]
    sizes = [v["size"] for v in per.values()]
    purs = [v["purity"] for v in per.values()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(sizes, purs, c=range(len(sizes)), cmap="viridis", s=80)
    for i, (c, v) in enumerate(per.items()):
        ax.annotate(CN[v["maj_class"]] if v["maj_class"] < len(CN) else str(v["maj_class"]),
                    (v["size"], v["purity"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("cluster size (# cached samples)")
    ax.set_ylabel("cluster purity")
    ax.set_title("Long-tail (ρ=100): cluster size vs. purity (tail classes are small/impure)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "lt_cluster_purity.png"), dpi=150)
    plt.close()
    print("saved lt_cluster_purity.png")


def main():
    make_overview_table()
    print()
    make_dynamic_table()
    plot_silhouette_curves()
    plot_attack_bar()
    plot_dynamic()
    plot_lt_per_class()
    print("\nDone. Tables ->", TBL, "| Figs ->", FIG)


if __name__ == "__main__":
    main()
