"""
analyze.py — Render the stealth-experiment tables and figures from
exp_stealth/outputs/results.json (+ the per-victim .npz arrays).

Tables (outputs/tables/*.txt):
    table1_monitoring.txt  — Part A: global/per-class accuracy monitoring
    table2_drift.txt       — Part B: prediction-drift statistics
    table3_perclass.txt    — per-class accuracy + per-class confidence drift
Figures (outputs/figs/*.png):
    per_class_acc.png      — per-class accuracy pre vs post (victim highlighted)
    confidence_dist.png    — confidence histograms clean vs attacked
    pred_hist.png          — predicted-class histogram (the detectable shift)
    drift_monitors.png     — which drift monitors fire vs their null threshold
    alert_sweep.png        — global-accuracy alert threshold sweep
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
EVAL = os.path.join(OUT, "eval")
CN = ('plane', 'car', 'bird', 'cat', 'deer',
      'dog', 'frog', 'horse', 'ship', 'truck')

C_CLEAN = "#1f5fa8"      # clean / benign blue
C_ATK = "#c0504d"        # attacked / victim red
C_VICTIM = "#c0504d"
C_NONV = "#4c9f70"       # non-victim green
C_NULL = "#9aa0a6"       # null threshold grey


def load_results():
    with open(os.path.join(OUT, "results.json")) as f:
        return json.load(f)


def load_npz(victim):
    p = os.path.join(EVAL, f"victim{victim}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def table_monitoring(res):
    lines = ["=" * 104,
             "TABLE 1 — Part A: validation-set performance monitoring (client holds labels)",
             "Global accuracy change is what a deployment KPI monitor sees; per-class",
             "audit is the finer-grained check the client *could* run.",
             "=" * 104]
    lines.append(f"{'Victim':<8}{'global pre':>12}{'post':>9}{'Δabs':>9}{'MP':>8}"
                 f"{'NVA-Drop':>10}{'alert>1%':>10}{'alert>2%':>10}{'cls|Δ|>1%':>11}")
    lines.append("-" * 104)
    for v in res["victims"]:
        A = v["partA_monitoring"]
        al = A["global_alert_fired_at_threshold"]
        pc = A["per_class_flagged_count_at_threshold"]
        lines.append(f"{A['victim_name']:<8}{A['global_acc_pre']:>11.2f}%"
                     f"{A['global_acc_post']:>8.2f}%{A['global_acc_abs_drop']:>+8.2f}%"
                     f"{A['misclass_prob']:>7.1f}%{A['nva_drop']:>9.2f}%"
                     f"{'YES' if al['1.00%'] else 'no':>10}"
                     f"{'YES' if al['2.00%'] else 'no':>10}"
                     f"{pc['1.00%']:>11}")
    lines.append("=" * 104)
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(TBL, "table1_monitoring.txt"), "w") as f:
        f.write(txt + "\n")


def table_drift(res):
    lines = ["=" * 116,
             "TABLE 2 — Part B: prediction-drift statistics (clean vs attacked outputs)",
             "Confidence/entropy drift = label-free output monitors (most common).",
             "Predicted-class PSI = label-free histogram monitor. null95 = 5%-FPR threshold.",
             "=" * 116]
    lines.append(f"{'Victim':<8}{'confKS all':>11}{'nonvictim':>10}{'victim':>8}"
                 f"{'entKS nv':>9}{'hist-PSI':>9}{'null95':>8}{'detect':>8}"
                 f"{'JS':>7}{'ECE cln':>9}{'ECE atk':>9}")
    lines.append("-" * 116)
    for v in res["victims"]:
        B = v["partB_drift"]
        cf = B["confidence"]; en = B["entropy"]; H = B["predicted_class_hist"]
        D = B["detection_thresholds"]; E = B["calibration_ece"]
        det = "YES" if D["psi_attack_detected"] else "no"
        lines.append(f"{v['victim_name']:<8}{cf['all']['ks_d']:>10.3f}"
                     f"{cf['nonvictim']['ks_d']:>10.3f}{cf['victim']['ks_d']:>8.3f}"
                     f"{en['nonvictim']['ks_d']:>9.3f}{H['psi']:>9.3f}"
                     f"{D['psi_null_95pct_5pctFPR']:>8.3f}{det:>8}"
                     f"{H['js_divergence']:>7.3f}{E['all_clean']:>9.3f}{E['all_atk']:>9.3f}")
    lines.append("=" * 116)
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(TBL, "table2_drift.txt"), "w") as f:
        f.write(txt + "\n")


def table_perclass(res):
    for v in res["victims"]:
        A = v["partA_monitoring"]; B = v["partB_drift"]
        victim = A["victim_name"]
        lines = ["=" * 78,
                 f"TABLE 3 — per-class detail for victim = {victim} "
                 f"(proxy = {A['proxy_name']})",
                 "=" * 78]
        lines.append(f"{'class':<8}{'acc pre':>9}{'acc post':>10}{'|Δ|':>8}"
                     f"{'conf-KS':>9}{'role':>12}")
        lines.append("-" * 78)
        ks_by = {d["name"]: d["ks_d"] for d in B["per_class_conf_ks"]}
        for c in CN:
            role = "VICTIM" if c == victim else ("proxy" if c == A["proxy_name"] else "")
            lines.append(f"{c:<8}{A['per_class_acc_pre'][c]:>8.1f}%"
                         f"{A['per_class_acc_post'][c]:>9.1f}%"
                         f"{A['per_class_abs_drop'][c]:>7.2f}%"
                         f"{ks_by[c]:>9.3f}{role:>12}")
        txt = "\n".join(lines)
        print(txt); print()
        with open(os.path.join(TBL, f"table3_perclass_{victim}.txt"), "w") as f:
            f.write(txt + "\n")


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def fig_per_class_acc(res):
    n = len(res["victims"])
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.2), squeeze=False)
    for ax, v in zip(axes[0], res["victims"]):
        A = v["partA_monitoring"]
        pre = [A["per_class_acc_pre"][c] for c in CN]
        post = [A["per_class_acc_post"][c] for c in CN]
        x = np.arange(len(CN)); w = 0.4
        cols = [C_VICTIM if c == A["victim_name"] else C_CLEAN for c in CN]
        ax.bar(x - w / 2, pre, w, color=cols, alpha=0.55, label="clean")
        ax.bar(x + w / 2, post, w, color=cols, label="after LASER")
        ax.set_xticks(x); ax.set_xticklabels(CN, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("per-class accuracy (%)"); ax.set_ylim(0, 105)
        ax.set_title(f"victim = {A['victim_name']} (→ {A['proxy_name']})\n"
                     f"global {A['global_acc_pre']:.1f}%→{A['global_acc_post']:.1f}%  "
                     f"(Δ{A['global_acc_abs_drop']:+.2f}%)")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "per_class_acc.png"), dpi=150)
    plt.close()
    print("saved per_class_acc.png")


def fig_confidence_dist(res):
    n = len(res["victims"])
    fig, axes = plt.subplots(2, n, figsize=(5.4 * n, 6.4), squeeze=False)
    for j, v in enumerate(res["victims"]):
        d = load_npz(v["victim"])
        if d is None:
            continue
        labels = d["labels"]; victim = v["victim"]
        mv = labels == victim; mnv = ~mv
        bins = np.linspace(0, 1, 31)
        # row 0: non-victim subset (stealth target)
        ax = axes[0][j]
        ax.hist(d["conf_clean"][mnv], bins, alpha=0.5, color=C_CLEAN,
                label="clean", density=True)
        ax.hist(d["conf_atk"][mnv], bins, alpha=0.5, color=C_ATK,
                label="after LASER", density=True)
        ks = v["partB_drift"]["confidence"]["nonvictim"]["ks_d"]
        ax.set_title(f"NON-victim samples (n={int(mnv.sum())})\n"
                     f"KS D={ks:.3f} (≈ sampling noise)")
        ax.set_xlabel("max softmax confidence"); ax.set_ylabel("density")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        # row 1: victim subset
        ax = axes[1][j]
        ax.hist(d["conf_clean"][mv], bins, alpha=0.5, color=C_CLEAN,
                label="clean", density=True)
        ax.hist(d["conf_atk"][mv], bins, alpha=0.5, color=C_ATK,
                label="after LASER", density=True)
        ks = v["partB_drift"]["confidence"]["victim"]["ks_d"]
        ax.set_title(f"VICTIM samples (n={int(mv.sum())})\nKS D={ks:.3f} (large)")
        ax.set_xlabel("max softmax confidence"); ax.set_ylabel("density")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Confidence distribution: clean vs LASER — unchanged on non-victims",
                 y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "confidence_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved confidence_dist.png")


def fig_pred_hist(res):
    n = len(res["victims"])
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.2), squeeze=False)
    for ax, v in zip(axes[0], res["victims"]):
        B = v["partB_drift"]; H = B["predicted_class_hist"]
        hc, ha = np.array(H["hist_clean"]), np.array(H["hist_atk"])
        x = np.arange(len(CN)); w = 0.4
        ax.bar(x - w / 2, hc, w, color=C_CLEAN, alpha=0.6, label="clean")
        ax.bar(x + w / 2, ha, w, color=C_ATK, alpha=0.9, label="after LASER")
        # mark victim & proxy
        vi = v["victim"]; pi = v["proxy"]
        ax.axvspan(vi - 0.5, vi + 0.5, color=C_VICTIM, alpha=0.08)
        ax.axvspan(pi - 0.5, pi + 0.5, color="#f0a830", alpha=0.12)
        ax.set_xticks(x); ax.set_xticklabels(CN, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("# predictions (test set)")
        ax.set_title(f"victim={CN[vi]} (red span) → proxy={CN[pi]} (orange)\n"
                     f"PSI={H['psi']:.2f} (null95="
                     f"{B['detection_thresholds']['psi_null_95pct_5pctFPR']:.3f}) "
                     f"{'DETECTED' if B['detection_thresholds']['psi_attack_detected'] else 'ok'}")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "pred_hist.png"), dpi=150)
    plt.close()
    print("saved pred_hist.png")


def fig_drift_monitors(res):
    """Grouped bars: attack drift signal vs the 5%-FPR null threshold, for each
    label-free output monitor (confidence-KS, entropy-KS, predicted-class PSI)."""
    monitors = ["conf-KS\n(all)", "conf-KS\n(non-victim)", "entropy-KS\n(non-victim)",
                "pred-class\nPSI"]
    n = len(res["victims"])
    fig, ax = plt.subplots(figsize=(7.5 + n, 4.6))
    x = np.arange(len(monitors)); w = 0.8 / max(n, 1)
    for i, v in enumerate(res["victims"]):
        B = v["partB_drift"]
        vals = [B["confidence"]["all"]["ks_d"],
                B["confidence"]["nonvictim"]["ks_d"],
                B["entropy"]["nonvictim"]["ks_d"],
                B["predicted_class_hist"]["psi"]]
        thr = [B["detection_thresholds"]["conf_ks_null_95pct_5pctFPR"]] * 3 + \
              [B["detection_thresholds"]["psi_null_95pct_5pctFPR"]]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w,
                      color=[C_NONV, C_NONV, C_NONV, C_ATK][i] if n > 1 else C_ATK,
                      alpha=0.85, label=f"victim={v['victim_name']}")
        for xi, (b, t) in enumerate(zip(bars, thr)):
            ax.hlines(t, b.get_x(), b.get_x() + b.get_width(),
                      colors=C_NULL, linestyles="--", linewidth=2)
    ax.set_xticks(x); ax.set_xticklabels(monitors, fontsize=9)
    ax.set_ylabel("drift statistic (bar) / 5%-FPR null threshold (dashed)")
    ax.set_title("Label-free output-drift monitors: attack signal vs detection threshold\n"
                 "(green ≈ below threshold = stealthy; red = the one that fires)")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "drift_monitors.png"), dpi=150)
    plt.close()
    print("saved drift_monitors.png")


def fig_alert_sweep(res):
    ths = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.8 / max(len(res["victims"]), 1)
    for i, v in enumerate(res["victims"]):
        A = v["partA_monitoring"]
        al = A["global_alert_fired_at_threshold"]
        fired = [1.0 if al[f"{t:.2f}%"] else 0.0 for t in ths]
        dg = A["global_acc_abs_drop"]
        x = np.arange(len(ths))
        ax.bar(x + i * width - 0.4 + width / 2, fired, width,
               alpha=0.7, label=f"{v['victim_name']} (Δ={dg:+.2f}%)")
    ax.set_xticks(np.arange(len(ths)))
    ax.set_xticklabels([f">{t:g}%" for t in ths])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["not flagged", "FLAGGED"])
    ax.set_title("Global-accuracy KPI monitor: does it flag LASER?\n"
                 "(flag fires only if |Δglobal acc| exceeds threshold)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "alert_sweep.png"), dpi=150)
    plt.close()
    print("saved alert_sweep.png")


def table_multiattack():
    p = os.path.join(OUT, "multiattack.json")
    if not os.path.exists(p):
        return
    with open(p) as f:
        m = json.load(f)
    lines = ["=" * 96,
             "TABLE 5 — Cumulative multi-class attack (cat -> truck -> bird, same server)",
             "Each row adds one victim on top of the previous. Visibility grows with the",
             "TOTAL victimized fraction (#attacked x ~10% on the balanced monitor).",
             "=" * 96]
    lines.append(f"{'#atk':>5}{'attacked':>22}{'global post':>13}{'Δglobal':>9}"
                 f"{'NVA(non-atk)':>13}{'pred-PSI':>9}{'collapsed':>10}"
                 f"{'confW(nv)':>10}")
    lines.append("-" * 96)
    for s in m["steps"]:
        atk = "+".join(s["attacked_classes"])
        coll = sum(1 for v in s["per_class_acc_post"].values() if v < 5.0)
        lines.append(f"{s['step']:>5}{atk:>22}{s['global_acc_post']:>12.2f}%"
                     f"{s['global_acc_drop']:>+8.2f}%{s['nva_drop_nonattacked']:>12.2f}%"
                     f"{s['predicted_psi']:>9.2f}{coll:>10}{s['conf_nonattacked']['wasserstein']:>10.3f}")
    lines.append("=" * 96)
    txt = "\n".join(lines)
    print(txt); print()
    with open(os.path.join(TBL, "table5_multiattack.txt"), "w") as f:
        f.write(txt + "\n")


def fig_multiattack():
    p = os.path.join(OUT, "multiattack.json")
    if not os.path.exists(p):
        return
    with open(p) as f:
        m = json.load(f)
    steps = m["steps"]
    x = [s["step"] for s in steps]
    drop = [s["global_acc_drop"] for s in steps]
    psi = [s["predicted_psi"] for s in steps]
    coll = [sum(1 for v in s["per_class_acc_post"].values() if v < 5.0) for s in steps]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.4))
    ax1.bar(np.array(x) - 0.15, drop, 0.3, color=C_ATK, label="Δglobal accuracy")
    ax1.set_xlabel("# classes attacked cumulatively")
    ax1.set_ylabel("global accuracy drop (%)", color=C_ATK)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"+{s['new_victim']}" for s in steps])
    ax2 = ax1.twinx()
    ax2.plot(x, psi, "-o", color="#1f5fa8", label="predicted-class PSI")
    ax2.set_ylabel("predicted-class PSI", color="#1f5fa8")
    for xi, c in zip(x, coll):
        ax1.annotate(f"{c} collapsed", (xi, drop[x.index(c)] if c else 0),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8,
                     color="#333")
    ax1.set_title("Cumulative multi-class attack: visibility grows with the victimized "
                  "fraction\n(each added class ≈ +9% Δglobal on the balanced monitor)")
    ax1.grid(axis="y", alpha=0.3)
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.95), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "multiattack.png"), dpi=150)
    plt.close()
    print("saved multiattack.png")


def table_baseline_drift():
    p = os.path.join(OUT, "baseline_drift.json")
    if not os.path.exists(p):
        return
    with open(p) as f:
        b = json.load(f)
    lines = ["=" * 92,
             "TABLE 6 — Contextualizing LASER's confidence drift vs benign model evolution",
             "Whole-monitoring-set drift for benign updates vs LASER's NON-VICTIM drift.",
             "=" * 92]
    lines.append(f"{'scenario':<34}{'conf Wass':>11}{'mean shift':>12}{'conf KS':>9}"
                 f"{'ent Wass':>10}{'Δacc':>8}{'PSI':>7}")
    lines.append("-" * 92)
    def row(name, d, acc_key="global_acc_change"):
        sh = d.get("conf_mean_shift", float("nan"))
        lines.append(f"{name:<34}{d['conf_wasserstein']:>11.4f}{sh:>+12.4f}"
                     f"{d['conf_ks']:>9.3f}{d['entropy_wasserstein']:>10.4f}"
                     f"{d.get(acc_key, 0.0):>+7.2f}%{d.get('pred_psi', 0.0):>7.3f}")
    row("B2 clean best<->last (noise)", b["B2_clean_best_vs_last"])
    row("B1 clean continued-training", b["B1_clean_continued_training"])
    for lz in b.get("LASER_nonvictim_drift", []):
        row(f"LASER {lz['victim']} (non-victim)", lz, acc_key=None)
    lines.append("=" * 92)
    txt = "\n".join(lines)
    print(txt); print()
    with open(os.path.join(TBL, "table6_baseline.txt"), "w") as f:
        f.write(txt + "\n")


def fig_baseline_drift():
    p = os.path.join(OUT, "baseline_drift.json")
    if not os.path.exists(p):
        return
    with open(p) as f:
        b = json.load(f)
    names, wass, shift = [], [], []
    names.append("B2: clean\nbest↔last"); wass.append(b["B2_clean_best_vs_last"]["conf_wasserstein"]); shift.append(abs(b["B2_clean_best_vs_last"]["conf_mean_shift"]))
    names.append("B1: clean\ncontinued train"); wass.append(b["B1_clean_continued_training"]["conf_wasserstein"]); shift.append(abs(b["B1_clean_continued_training"]["conf_mean_shift"]))
    for lz in b.get("LASER_nonvictim_drift", []):
        names.append(f"LASER {lz['victim']}\n(non-victim)"); wass.append(lz["conf_wasserstein"]); shift.append(abs(lz["conf_mean_shift"]))
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = ["#4c9f70", "#4c9f70"] + ["#c0504d"] * (len(names) - 2)
    ax1.bar(x, wass, color=cols, alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("confidence Wasserstein distance")
    ax1.set_title("Confidence-distribution drift magnitude")
    ax1.grid(axis="y", alpha=0.3)
    ax2.bar(x, shift, color=cols, alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("|mean-confidence shift|")
    ax2.set_title("Mean confidence shift (LASER ≈ 0.04)")
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Is LASER's 0.04 confidence shift abnormal?  (green = benign model evolution)",
                 y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "baseline_drift.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved baseline_drift.png")


def main():
    # Part C (prevalence sweep) resamples cached predictions and appends to
    # results.json, so run it first.
    from exp_stealth import prevalence
    prevalence.run()
    print()
    res = load_results()
    table_monitoring(res)
    print()
    table_drift(res)
    print()
    table_perclass(res)
    fig_per_class_acc(res)
    fig_confidence_dist(res)
    fig_pred_hist(res)
    fig_drift_monitors(res)
    fig_alert_sweep(res)
    # follow-up experiments (read their JSON if present)
    table_multiattack(); fig_multiattack()
    table_baseline_drift(); fig_baseline_drift()
    print(f"\nDone. Tables -> {TBL} | Figs -> {FIG}")


if __name__ == "__main__":
    main()
