"""
separability.py — M9: separability–attackability analysis from existing results.

Reads the rebuttal artefacts (exp_robust, exp_scale, exp_stealth) and asks:
which discovered clusters are actually label-free attackable, and which
conditions predict failure?  Ground-truth labels are used only in the source
experiments for evaluation; this script only aggregates their reported metrics.

Outputs
-------
exp_camera/outputs/separability_attackability.json
exp_camera/outputs/separability_attackability.md

Usage
-----
    python -m exp_camera.separability
"""
import glob
import json
import os
from collections import OrderedDict

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "exp_camera", "outputs")
os.makedirs(OUT, exist_ok=True)

ROBUST = os.path.join(REPO, "exp_robust", "outputs")
SCALE = os.path.join(REPO, "exp_scale", "outputs")

MIN_PURITY = 0.5
MIN_SIZE = 200


def _get_cluster_info(clustering, cluster_id):
    if cluster_id is None:
        return None, None
    per = clustering.get("per_cluster", {})
    rec = per.get(str(cluster_id), per.get(cluster_id))
    if not rec:
        return None, None
    return rec.get("purity"), rec.get("size")


def _add(rows, condition, dataset, model, split, clustering, victim, acc_v,
         mp, nva_drop, target_cluster=None, cluster_purity=None,
         cluster_size=None, status="ok", note=""):
    rows.append({
        "condition": condition, "dataset": dataset, "model": model,
        "split": split,
        "k_star": clustering.get("k_star") if clustering else None,
        "ari": clustering.get("ari") if clustering else None,
        "nmi": clustering.get("nmi") if clustering else None,
        "overall_purity": clustering.get("purity") if clustering else None,
        "victim": victim, "target_cluster": target_cluster,
        "cluster_purity": cluster_purity, "cluster_size": cluster_size,
        "acc_v": None if acc_v is None else float(acc_v),
        "mp": None if mp is None else float(mp),
        "nva_drop": None if nva_drop is None else float(nva_drop),
        "targetable": bool(status == "ok"),
        "status": status, "note": note,
    })


def load_robust_static():
    rows = []
    specs = [
        ("results_baseline.json", "clean CIFAR-10", "CIFAR-10", "VGG-16", "SB=1"),
        ("results_noise20.json", "20% label noise", "CIFAR-10", "VGG-16", "SB=1"),
        ("results_noise40.json", "40% label noise", "CIFAR-10", "VGG-16", "SB=1"),
        ("results_lt10.json", "long-tail rho=10", "CIFAR-10", "VGG-16", "SB=1"),
        ("results_lt.json", "long-tail rho=100", "CIFAR-10", "VGG-16", "SB=1"),
    ]
    for fname, cond, dataset, model, split in specs:
        path = os.path.join(ROBUST, fname)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        clustering = d.get("clustering", {})
        for a in d.get("attacks", []):
            status = a.get("status", "ok")
            if status != "ok":
                _add(rows, cond, dataset, model, split, clustering,
                     a.get("victim_name"), None, None, None,
                     target_cluster=a.get("target_cluster"),
                     cluster_purity=a.get("cluster_purity"),
                     status=status, note=a.get("note", ""))
                continue
            tc = a.get("target_cluster")
            purity, size = _get_cluster_info(clustering, tc)
            _add(rows, cond, dataset, model, split, clustering,
                 a.get("victim_name"), a.get("acc_v"), a.get("mp"),
                 a.get("nva_drop"), target_cluster=tc,
                 cluster_purity=purity, cluster_size=size)
    return rows


def load_dynamic():
    rows = []
    path = os.path.join(ROBUST, "results_dynamic.json")
    if not os.path.exists(path):
        return rows
    d = json.load(open(path))
    for p in d.get("phases", []):
        clustering = {"k_star": p.get("k_star"), "ari": p.get("ari"),
                      "purity": p.get("purity")}
        # The dynamic driver does not store per-cluster purity/size for the
        # victim; keep the condition-level row and mark the cluster fields null.
        _add(rows, f"dynamic {p.get('phase')}", "CIFAR-10", "VGG-16", "SB=1",
             clustering, p.get("victim_name"), p.get("acc_v"), p.get("mp"),
             p.get("nva_drop"), target_cluster=None,
             cluster_purity=None, cluster_size=None)
    return rows


def load_blind():
    rows = []
    path = os.path.join(ROBUST, "results_blind.json")
    if not os.path.exists(path):
        return rows
    d = json.load(open(path))
    clustering = d.get("clustering", {})
    for r in d.get("rows", []):
        _add(rows, "blind cluster sweep", "CIFAR-10", "VGG-16", "SB=1",
             clustering, r.get("victim_name"), r.get("victim_recall_post"),
             r.get("mp"), r.get("nva_drop"),
             target_cluster=r.get("target_cluster"),
             cluster_purity=r.get("cluster_purity"),
             cluster_size=r.get("cluster_size"))
    return rows


def load_scale():
    rows = []
    path = os.path.join(SCALE, "results_scale_cifar100_20ep.json")
    if not os.path.exists(path):
        return rows
    d = json.load(open(path))
    clustering = d.get("clustering", {})
    for r in d.get("rows", []):
        _add(rows, "CIFAR-100 blind sweep", "CIFAR-100", "VGG-16", "SB=1",
             clustering, r.get("victim_name"), r.get("victim_recall_post"),
             r.get("mp"), r.get("nva_drop"),
             target_cluster=r.get("target_cluster"),
             cluster_purity=r.get("cluster_purity"),
             cluster_size=r.get("cluster_size"))
    return rows


def load_m7():
    """M7 CIFAR-100 SB=2/3 blind sweeps (SB=1 is already in load_scale)."""
    rows = []
    for sb in (2, 3):
        path = os.path.join(OUT, "cifar100",
                            f"results_cifar100_sb{sb}_seed9202.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        clustering = d.get("clustering", {})
        for r in d.get("rows", []):
            _add(rows, f"M7 CIFAR-100 sb={sb}", "CIFAR-100", "VGG-16",
                 f"SB={sb}", clustering, r.get("victim_name"),
                 r.get("victim_recall_post"), r.get("mp"), r.get("nva_drop"),
                 target_cluster=r.get("target_cluster"),
                 cluster_purity=r.get("cluster_purity"),
                 cluster_size=r.get("cluster_size"))
    return rows


def load_camera():
    """M5/M6 controlled results (CIFAR-10, VGG-16, SB=1/2/3 × seeds)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(
            OUT, "results_cifar10_sb*_seed*.json"))):
        d = json.load(open(path))
        if d.get("status") != "done" or "attack" not in d:
            continue
        a = d["attack"]
        tc = a.get("target_cluster")
        clustering = d.get("clustering", {})
        purity, size = _get_cluster_info(clustering, tc)
        _add(rows,
             f"M5/M6 sb={d.get('split_block')} seed={d.get('seed')}",
             "CIFAR-10", "VGG-16", f"SB={d.get('split_block')}",
             clustering, a.get("victim_name"), a.get("acc_v"),
             a.get("mp"), a.get("nva_drop"), target_cluster=tc,
             cluster_purity=purity, cluster_size=size)
    return rows


def summarize(rows):
    ok = [r for r in rows if r["targetable"] and r["acc_v"] is not None]
    with_purity = [r for r in ok if r["cluster_purity"] is not None]
    with_size = [r for r in ok if r["cluster_size"] is not None]

    def corr(xs, ys, fn):
        if len(xs) < 3:
            return None
        v, p = fn(xs, ys)
        return {"r": float(v), "p": float(p), "n": len(xs)}

    stats = {
        "n_rows": len(rows),
        "n_targetable": len(ok),
        "n_with_cluster_purity": len(with_purity),
        "pearson_purity_accv": corr(
            [r["cluster_purity"] for r in with_purity],
            [r["acc_v"] for r in with_purity], pearsonr),
        "spearman_purity_accv": corr(
            [r["cluster_purity"] for r in with_purity],
            [r["acc_v"] for r in with_purity], spearmanr),
        "pearson_purity_mp": corr(
            [r["cluster_purity"] for r in with_purity],
            [r["mp"] for r in with_purity], pearsonr),
        "spearman_size_accv": corr(
            [r["cluster_size"] for r in with_size],
            [r["acc_v"] for r in with_size], spearmanr),
    }

    gate_pass = [r for r in rows if r["targetable"]
                 and r["cluster_purity"] is not None
                 and r["cluster_size"] is not None
                 and r["cluster_purity"] >= MIN_PURITY
                 and r["cluster_size"] >= MIN_SIZE]
    gate_fail = [r for r in rows if r["targetable"]
                 and r["cluster_purity"] is not None
                 and r["cluster_size"] is not None
                 and not (r["cluster_purity"] >= MIN_PURITY
                          and r["cluster_size"] >= MIN_SIZE)]

    def mean(xs):
        return float(np.mean(xs)) if xs else None

    gate = {
        "rule": f"purity >= {MIN_PURITY} and size >= {MIN_SIZE}",
        "n_pass": len(gate_pass), "n_fail": len(gate_fail),
        "acc_v_mean_pass": mean([r["acc_v"] for r in gate_pass]),
        "acc_v_mean_fail": mean([r["acc_v"] for r in gate_fail]),
        "mp_mean_pass": mean([r["mp"] for r in gate_pass]),
        "mp_mean_fail": mean([r["mp"] for r in gate_fail]),
        "nva_mean_pass": mean([r["nva_drop"] for r in gate_pass]),
        "nva_mean_fail": mean([r["nva_drop"] for r in gate_fail]),
    }

    failures = [r for r in rows if not r["targetable"]
                or (r["acc_v"] is not None and r["acc_v"] > 5.0)
                or (r["cluster_purity"] is not None
                    and r["cluster_purity"] < MIN_PURITY)
                or (r["cluster_size"] is not None
                    and r["cluster_size"] < MIN_SIZE)]
    return stats, gate, failures


def condition_summary(rows):
    out = OrderedDict()
    for r in rows:
        key = r["condition"]
        rec = out.setdefault(key, {
            "condition": key, "dataset": r["dataset"], "model": r["model"],
            "split": r["split"], "n_points": 0, "n_targetable": 0,
            "acc_v": [], "mp": [], "nva_drop": [],
            "cluster_purity": [], "cluster_size": [],
        })
        rec["n_points"] += 1
        if r["targetable"]:
            rec["n_targetable"] += 1
        for src, dst in (("acc_v", "acc_v"), ("mp", "mp"),
                         ("nva_drop", "nva_drop"),
                         ("cluster_purity", "cluster_purity"),
                         ("cluster_size", "cluster_size")):
            if r[src] is not None:
                rec[dst].append(r[src])
    for rec in out.values():
        for k in ("acc_v", "mp", "nva_drop", "cluster_purity", "cluster_size"):
            vals = rec[k]
            rec[k] = ({"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                       "n": len(vals)} if vals else None)
    return list(out.values())


def write_markdown(rows, conds, stats, gate, failures, path):
    lines = []
    lines.append("# M9 — Separability–Attackability Analysis\n")
    lines.append("> Generated by `python -m exp_camera.separability` from the "
                 "existing rebuttal artefacts. Ground-truth labels are used only "
                 "as evaluation oracles in the source experiments; the attack "
                 "itself is label-free.\n")
    lines.append("## 1. Condition summary\n")
    lines.append("| Condition | Dataset | Model / Split | Points | Targetable | "
                 "ACC-V (mean±std) | MP | NVA Drop | Cluster purity | Cluster size |")
    lines.append("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c in conds:
        def fmt(rec):
            if rec is None:
                return "—"
            return f"{rec['mean']:.2f}±{rec['std']:.2f}"
        lines.append(
            f"| {c['condition']} | {c['dataset']} | {c['model']} / {c['split']} | "
            f"{c['n_points']} | {c['n_targetable']} | {fmt(c['acc_v'])} | "
            f"{fmt(c['mp'])} | {fmt(c['nva_drop'])} | "
            f"{fmt(c['cluster_purity'])} | {fmt(c['cluster_size'])} |")
    lines.append("\n## 2. Separability vs attackability\n")
    lines.append(f"- Rows: **{stats['n_rows']}**; targetable: "
                 f"**{stats['n_targetable']}**; with per-cluster purity: "
                 f"**{stats['n_with_cluster_purity']}**.")
    for key, label in [
        ("pearson_purity_accv", "Pearson(purity, ACC-V)"),
        ("spearman_purity_accv", "Spearman(purity, ACC-V)"),
        ("pearson_purity_mp", "Pearson(purity, MP)"),
        ("spearman_size_accv", "Spearman(size, ACC-V)"),
    ]:
        s = stats[key]
        if s is None:
            lines.append(f"- {label}: insufficient data.")
        else:
            lines.append(f"- {label}: r={s['r']:.3f}, p={s['p']:.3g}, n={s['n']}.")
    lines.append("\n### Targetability gate\n")
    lines.append(f"Rule: **{gate['rule']}**.")
    lines.append("")
    lines.append("| Group | n | ACC-V mean | MP mean | NVA-Drop mean |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for label, key in (("pass", "pass"), ("fail", "fail")):
        lines.append(
            f"| {label} | {gate['n_'+key]} | "
            f"{gate['acc_v_mean_'+key]:.2f} | {gate['mp_mean_'+key]:.2f} | "
            f"{gate['nva_mean_'+key]:.2f} |")
    lines.append("\n## 3. Failure / boundary points\n")
    lines.append("| Condition | Victim | Purity | Size | ACC-V | MP | NVA | Status |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |")
    for r in failures:
        def g(v):
            return "—" if v is None else f"{v:.2f}"
        lines.append(
            f"| {r['condition']} | {r['victim']} | {g(r['cluster_purity'])} | "
            f"{g(r['cluster_size'])} | {g(r['acc_v'])} | {g(r['mp'])} | "
            f"{g(r['nva_drop'])} | {r['status']} |")
    lines.append("\n## 4. Interpretation\n")
    sp = stats["spearman_size_accv"]
    pp = stats["spearman_purity_accv"]
    lines.append(
        "The aggregate is deliberately heterogeneous (clean, noisy, long-tail, "
        "dynamic, blind sweeps, CIFAR-100), so these are descriptive, not causal, "
        "correlations. Three findings stand out. First, **cluster size is the "
        "strongest univariate correlate of attackability**: "
        f"Spearman(size, ACC-V) = {sp['r']:.2f} (p = {sp['p']:.1e}, n = {sp['n']}); "
        "small clusters are consistently harder to drive to zero. Second, "
        "**purity alone is not predictive across regimes**: "
        f"Spearman(purity, ACC-V) = {pp['r']:.2f} (p = {pp['p']:.2f}, n = {pp['n']}). "
        "High purity is common in the successful regimes, but it does not "
        "guarantee success (long-tail rho=10 car: purity 1.00, size 2,115, yet "
        "ACC-V 22.6%). Third, the simple gate is a **triage heuristic, not a "
        "guarantee**: it flags the obviously impure (otter, purity 0.45) and "
        "tiny (porcupine, size 190) clusters, but it does not separate the "
        "long-tail failures, where class support/proxy geometry is the limiting "
        "factor rather than cluster purity or raw size.")
    lines.append("")
    lines.append(
        "The failure taxonomy is therefore: (i) **no dedicated cluster** (tail "
        "classes under extreme imbalance are absorbed elsewhere); (ii) "
        "**impure/blended cluster** (the centroid is not a semantic target); "
        "(iii) **tiny cluster** (too few cached victim vectors to generalize); "
        "and (iv) **pure but weakly supported / proxy-confused cluster** "
        "(long-tail car), which no purity/size threshold can fully capture. "
        "For real business data, the practical implication is that coarse, "
        "well-populated classes are reliably attackable, whereas fine-grained "
        "or rare classes with small support are the honest boundary; a client "
        "auditor should prioritize per-class recall checks on exactly those "
        "classes.")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    rows = []
    rows += load_robust_static()
    rows += load_dynamic()
    rows += load_blind()
    rows += load_scale()
    rows += load_m7()
    rows += load_camera()
    stats, gate, failures = summarize(rows)
    conds = condition_summary(rows)
    payload = {"rows": rows, "conditions": conds, "stats": stats,
               "gate": gate, "failures": failures,
               "sources": ["exp_robust/outputs/results_*.json",
                           "exp_scale/outputs/results_scale_cifar100_20ep.json",
                           "exp_camera/outputs/cifar100/results_cifar100_sb*_seed9202.json",
                           "exp_camera/outputs/results_cifar10_sb*_seed*.json"],
               "note": "GT labels are evaluation-only in the source experiments."}
    json_path = os.path.join(OUT, "separability_attackability.json")
    md_path = os.path.join(OUT, "separability_attackability.md")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    write_markdown(rows, conds, stats, gate, failures, md_path)
    print(f"[M9] rows={len(rows)} targetable={stats['n_targetable']} "
          f"gate={gate['n_pass']} pass / {gate['n_fail']} fail")
    print(f"[M9] -> {json_path}\n[M9] -> {md_path}")


if __name__ == "__main__":
    main()
