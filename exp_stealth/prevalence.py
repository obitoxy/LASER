"""
prevalence.py — Part C of the stealth study: how LASER's visibility to GLOBAL
accuracy / predicted-class monitoring scales with the victim's prevalence in the
monitored population.

Motivation. The global-accuracy drop on a *balanced* monitor is dominated by the
victim class collapsing to 0 % (Δglobal ≈ victim_prevalence × victim_acc). The
balanced CIFAR-10 test set (every class 10 %) is the WORST case. Real deployment
monitoring runs on live traffic, which is rarely balanced, and a targeted attack
in practice usually targets a MINORITY / sensitive class. This module sweeps the
victim prevalence p and reports Δglobal-accuracy and predicted-class PSI.

No model is re-run: we reweight/resample the per-sample clean vs attacked
predictions already cached in outputs/eval/victim*.npz.
"""
import os
import json
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
TBL = os.path.join(OUT, "tables"); os.makedirs(TBL, exist_ok=True)
FIG = os.path.join(OUT, "figs"); os.makedirs(FIG, exist_ok=True)
EVAL = os.path.join(OUT, "eval")
CN = ('plane', 'car', 'bird', 'cat', 'deer',
      'dog', 'frog', 'horse', 'ship', 'truck')

PREVS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]   # victim fraction of monitor
N_MON = 10000                                      # monitor size (matches test set)
SEEDS = list(range(20))
ALERT_THRS = [0.5, 1.0, 2.0, 5.0]


def _global_acc(preds, labels):
    return 100.0 * float((preds == labels).mean())


def _psi(a, b, nc=10):
    ha = np.bincount(a, minlength=nc).astype(np.float64)
    hb = np.bincount(b, minlength=nc).astype(np.float64)
    pa = ha / ha.sum(); pb = hb / hb.sum()
    pa = np.clip(pa, 1e-12, None); pb = np.clip(pb, 1e-12, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def sweep_victim(victim, data):
    """For one victim, sweep prevalence. Returns list of dicts."""
    labels = data["labels"]
    preds_c, preds_a = data["preds_clean"], data["preds_atk"]
    mv = labels == victim
    mnv = ~mv
    vic_idx = np.where(mv)[0]
    nv_idx = np.where(mnv)[0]
    rows = []
    for p in PREVS:
        k = int(round(p * N_MON))                       # victim count in monitor
        k = max(k, 1)
        drops, psis, gpre, gpost = [], [], [], []
        for s in SEEDS:
            rng = np.random.RandomState(1000 + s)
            # victim samples (without replacement if enough, else with)
            vpick = rng.choice(vic_idx, size=k, replace=(k > len(vic_idx)))
            nvpick = rng.choice(nv_idx, size=N_MON - k, replace=(N_MON - k > len(nv_idx)))
            idx = np.concatenate([vpick, nvpick])
            gc, ga = _global_acc(preds_c[idx], labels[idx]), _global_acc(preds_a[idx], labels[idx])
            drops.append(gc - ga); gpre.append(gc); gpost.append(ga)
            psis.append(_psi(preds_c[idx], preds_a[idx]))
        drop = float(np.mean(drops))
        row = {"prevalence": p, "victim_n_in_monitor": k,
               "global_acc_pre": float(np.mean(gpre)),
               "global_acc_post": float(np.mean(gpost)),
               "global_acc_drop": drop,
               "pred_psi": float(np.mean(psis)),
               "alert_fired_at": {f"{t:.1f}%": bool(drop > t) for t in ALERT_THRS}}
        rows.append(row)
    return rows


def run():
    with open(os.path.join(OUT, "results.json")) as f:
        res = json.load(f)
    sweep = {}
    for v in res["victims"]:
        vic = v["victim"]
        p = os.path.join(EVAL, f"victim{vic}.npz")
        if not os.path.exists(p):
            print(f"[prevalence] missing {p}, skip"); continue
        d = np.load(p)
        data = {k: d[k] for k in d.files}
        sweep[v["victim_name"]] = sweep_victim(vic, data)

    # ---- table ----
    lines = ["=" * 84,
             "TABLE 4 — Part C: global-monitoring visibility vs victim prevalence",
             "Monitor size = 10000. Δglobal and pred-class PSI as the victim's share of",
             "the monitored population shrinks (balanced benchmark = 10 %, worst case).",
             "=" * 84]
    for vname, rows in sweep.items():
        lines.append(f"\nvictim = {vname}")
        lines.append(f"{'prev':>7}{'vic#':>7}{'global pre':>12}{'post':>9}{'Δglobal':>9}"
                     f"{'pred-PSI':>10}{'alert>1%':>10}{'alert>2%':>10}")
        lines.append("-" * 84)
        for r in rows:
            a = r["alert_fired_at"]
            lines.append(f"{r['prevalence']:>6.1%}{r['victim_n_in_monitor']:>7}"
                         f"{r['global_acc_pre']:>11.2f}%{r['global_acc_post']:>8.2f}%"
                         f"{r['global_acc_drop']:>+8.2f}%{r['pred_psi']:>10.3f}"
                         f"{'YES' if a['1.0%'] else 'no':>10}"
                         f"{'YES' if a['2.0%'] else 'no':>10}")
    lines.append("=" * 84)
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(TBL, "table4_prevalence.txt"), "w") as f:
        f.write(txt + "\n")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#c0504d", "#1f5fa8"]
    for (vname, rows), col in zip(sweep.items(), colors):
        ps = [r["prevalence"] * 100 for r in rows]
        ax1.plot(ps, [r["global_acc_drop"] for r in rows], "-o", color=col, label=vname)
        ax2.plot(ps, [r["pred_psi"] for r in rows], "-o", color=col, label=vname)
    for t in (1.0, 2.0):
        ax1.axhline(t, color="#9aa0a6", ls="--", lw=1)
    ax1.text(0.2, 1.05, "1% KPI threshold", color="#9aa0a6", fontsize=8)
    ax1.axhline(0.1, color="#4c9f70", ls=":", lw=1)
    ax1.text(0.2, 0.12, "0.1% (sampling noise)", color="#4c9f70", fontsize=8)
    ax1.set_xscale("log"); ax1.set_xlabel("victim prevalence in monitor (%) [log]")
    ax1.set_ylabel("global accuracy drop (%)"); ax1.set_title("Global-accuracy KPI monitor")
    ax1.grid(alpha=0.3, which="both"); ax1.legend(fontsize=8)
    ax2.axhline(0.1, color="#9aa0a6", ls="--", lw=1)
    ax2.text(0.2, 0.11, "PSI 0.1 (minor drift)", color="#9aa0a6", fontsize=8)
    ax2.set_xscale("log"); ax2.set_xlabel("victim prevalence in monitor (%) [log]")
    ax2.set_ylabel("predicted-class PSI"); ax2.set_title("Predicted-class distribution drift")
    ax2.grid(alpha=0.3, which="both"); ax2.legend(fontsize=8)
    fig.suptitle("LASER visibility scales with victim prevalence — stealthy against global "
                 "KPI monitoring for minority victims", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "prevalence.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved prevalence.png")

    # persist into results.json
    res["partC_prevalence"] = sweep
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("[prevalence] appended partC_prevalence to results.json")


if __name__ == "__main__":
    run()
