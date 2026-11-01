"""
lib.py — Stealth / client-side detection evaluation for LASER (RAID 2026
rebuttal, reviewer 296B: "why would the sharp semantic change not be visible
through client-side validation, prediction drift, or deployment monitoring?").

Two evaluation families, both computed on the CIFAR-10 **test set**, which here
plays the role of the *client's reserved, labeled validation / monitoring set*
(in USL the client holds the labels and the classifier head, so it can always
run this kind of audit):

  * Part A — Validation-set performance monitoring:
        global accuracy (pre/post), per-class accuracy, NVA-Drop, ACC-V, MP,
        and a global-accuracy alert-threshold sweep (does a KPI monitor fire?).

  * Part B — Prediction drift (output-distribution monitoring):
        confidence (max-softmax) and entropy distributions (KS + Wasserstein),
        predicted-class histogram drift (PSI + JS divergence), calibration
        (ECE), a bootstrap null for the PSI / KS detection threshold (FPR), and
        per-class confidence drift (which classes' output distributions moved).

The *attacked* model is produced by the **exact paper attack code path**
(`exp_robust.lib.run_laser_attack` -> `src.attack.attacks_dataloader1`); this
module only adds the detection-side measurement on top, so the stealth numbers
are directly comparable to the paper's NVA-Drop / MP.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from scipy import stats as st

import exp_robust.lib as L
from exp_robust.lib import NUM_CLASSES, CLASS_NAMES

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
STEALTH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(STEALTH_DIR)
OUT = os.path.join(STEALTH_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)
TBL = os.path.join(OUT, "tables"); os.makedirs(TBL, exist_ok=True)
FIG = os.path.join(OUT, "figs"); os.makedirs(FIG, exist_ok=True)

# the clean baseline model (VGG-16, SB=1) lives under exp_robust/outputs/models
MODEL_DIR = os.path.join(REPO, "exp_robust", "outputs", "models", "baseline")

EPS = 1e-12
DEVICE = L.DEVICE


# ======================================================================
# 1. MODEL LOADING + FORWARD (clean vs attacked server)
# ======================================================================
def load_split(model_dir=MODEL_DIR, server_ckpt=None):
    """Load net1 + net3 (always clean) and a server. If server_ckpt is None the
    clean (pre-attack) server is loaded; otherwise the attacked checkpoint."""
    net1, server, net3 = L.create_models()
    net1.load_state_dict(torch.load(os.path.join(model_dir, "net1_nor_last.pt")))
    net3.load_state_dict(torch.load(os.path.join(model_dir, "net3_nor_last.pt")))
    if server_ckpt is None:
        server.load_state_dict(torch.load(os.path.join(model_dir, "server_nor_last.pt")))
    else:
        server.load_state_dict(torch.load(server_ckpt))
    net1.eval(); server.eval(); net3.eval()
    return net1, server, net3


def test_loader(batch_size=512, num_workers=8):
    """Standard clean balanced CIFAR-10 test set (the client's monitoring set)."""
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=L.get_transforms(False))
    return DataLoader(testset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def forward_outputs(net1, server, net3, loader):
    """Forward `loader` through net1->server->net3. Returns:
        probs   (N, C) softmax probabilities (fp32)
        preds   (N,)   argmax class
        conf    (N,)   max softmax confidence
        entropy (N,)   predictive entropy (nats)
        labels  (N,)   ground truth
    Eval uses bf16 autocast (matches the paper's eval path)."""
    probs, preds, confs, ents, labels = [], [], [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = net3(server(net1(x)))
        p = F.softmax(out.float(), dim=1)
        preds.append(p.argmax(1).cpu().numpy())
        confs.append(p.max(1)[0].cpu().numpy())
        ents.append((-(p * torch.log(p + EPS)).sum(1)).cpu().numpy())
        labels.append(y.numpy())
        probs.append(p.cpu().numpy())
    return (np.concatenate(probs), np.concatenate(preds),
            np.concatenate(confs), np.concatenate(ents),
            np.concatenate(labels))


# ======================================================================
# 2. ACCURACY / MONITORING METRICS  (Part A)
# ======================================================================
def per_class_acc(preds, labels, num_classes=NUM_CLASSES):
    pc = np.zeros(num_classes); pt = np.zeros(num_classes)
    for c in range(num_classes):
        m = labels == c
        pt[c] = m.sum()
        pc[c] = (preds[m] == c).sum()
    return 100.0 * pc / np.maximum(pt, 1), pt


def global_acc(preds, labels):
    return 100.0 * float((preds == labels).mean())


def nva_drop(per_pre, per_post, victim):
    others = [c for c in range(NUM_CLASSES) if c != victim]
    pre = np.mean([per_pre[c] for c in others])
    post = np.mean([per_post[c] for c in others])
    return float(pre - post), float(pre), float(post)


def misclass_prob(preds, labels, victim, proxy):
    mv = labels == victim
    if mv.sum() == 0:
        return 0.0
    return 100.0 * float((preds[mv] == proxy).mean())


# ======================================================================
# 3. PREDICTION-DRIFT METRICS  (Part B)
# ======================================================================
def ks_stats(a, b):
    """Two-sample Kolmogorov–Smirnov: returns (D statistic, p-value)."""
    d, p = st.ks_2samp(a, b)
    return float(d), float(p)


def wasserstein(a, b):
    return float(st.wasserstein_distance(a, b))


def class_hist(preds, num_classes=NUM_CLASSES):
    h = np.bincount(preds, minlength=num_classes).astype(np.float64)
    return h


def psi(hist_ref, hist_cur):
    """Population Stability Index over class-histogram proportions.
    Conventional thresholds: <0.1 negligible, 0.1-0.25 minor, >0.25 major."""
    ref = hist_ref / max(hist_ref.sum(), 1.0)
    cur = hist_cur / max(hist_cur.sum(), 1.0)
    ref = np.clip(ref, EPS, None); cur = np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def js_divergence(hist_ref, hist_cur):
    ref = hist_ref / max(hist_ref.sum(), 1.0)
    cur = hist_cur / max(hist_cur.sum(), 1.0)
    m = 0.5 * (ref + cur)
    def kl(a, b):
        a = np.clip(a, EPS, None); b = np.clip(b, EPS, None)
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(ref, m) + 0.5 * kl(cur, m)


def chi_square(hist_ref, hist_cur):
    """Chi-square statistic of the predicted-class histogram (goodness of shift)."""
    ref = hist_ref / max(hist_ref.sum(), 1.0) * max(hist_cur.sum(), 1.0)
    ref = np.clip(ref, 1e-9, None)
    return float(st.chisquare(hist_cur, ref).statistic)


def expected_calibration_error(probs, labels, n_bins=15):
    confs = probs.max(1); preds = probs.argmax(1)
    acc = (preds == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0; n = len(labels)
    for i in range(n_bins):
        if i == 0:
            m = (confs >= edges[i]) & (confs <= edges[i + 1])
        else:
            m = (confs > edges[i]) & (confs <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(acc[m].mean() - confs[m].mean())
    return float(ece)


def mean_abs_logit_gap(probs):
    """Mean |p1 - p2| between the top-two class probabilities — a sharpness
    proxy that should be stable if the attack preserves confidence geometry."""
    sp = np.sort(probs, axis=1)
    return float(np.mean(sp[:, -1] - sp[:, -2]))


# ----------------------------------------------------------------------
# Bootstrap null for detection thresholds (sets FPR on a clean model).
# ----------------------------------------------------------------------
def bootstrap_psi_null(preds_clean, n_boot=500, size=None, seed=0):
    """Null distribution of the predicted-class PSI under H0 (model unchanged):
    repeatedly split the clean predictions into two random halves and compute
    PSI between their histograms. A monitor's 95th-percentile threshold => 5%
    FPR. Returns sorted array of null PSI values."""
    rng = np.random.RandomState(seed)
    n = len(preds_clean)
    size = size or n // 2
    null = []
    for _ in range(n_boot):
        idx = rng.permutation(n)
        a, b = preds_clean[idx[:size]], preds_clean[idx[size:2 * size]]
        null.append(psi(class_hist(a), class_hist(b)))
    return np.array(sorted(null))


def bootstrap_ks_null(values_clean, n_boot=500, size=None, seed=0):
    """Null distribution of the two-sample KS D statistic under H0: split the
    clean per-sample values (confidence or entropy) into two random halves."""
    rng = np.random.RandomState(seed)
    n = len(values_clean)
    size = size or n // 2
    null = []
    for _ in range(n_boot):
        idx = rng.permutation(n)
        a, b = values_clean[idx[:size]], values_clean[idx[size:2 * size]]
        null.append(ks_stats(a, b)[0])
    return np.array(sorted(null))


def per_class_conf_ks(confs_clean, confs_atk, labels, num_classes=NUM_CLASSES):
    """Per-true-class KS D on the confidence distribution (clean vs attacked).
    Non-victim classes should be ~0 (stealth); the victim class large."""
    out = []
    for c in range(num_classes):
        m = labels == c
        if m.sum() < 5:
            out.append({"class": int(c), "name": CLASS_NAMES[c], "n": int(m.sum()),
                        "ks_d": 0.0, "ks_p": 1.0})
        else:
            d, p = ks_stats(confs_clean[m], confs_atk[m])
            out.append({"class": int(c), "name": CLASS_NAMES[c], "n": int(m.sum()),
                        "ks_d": float(d), "ks_p": float(p)})
    return out


# ======================================================================
# 4. END-TO-END STEALTH EVALUATION (clean vs one attacked checkpoint)
# ======================================================================
def evaluate_stealth(victim, proxy, server_ckpt, loader=None,
                     n_boot=500, seed=0):
    """Run the full Part A + Part B comparison between the clean model and the
    attacked server checkpoint `server_ckpt`. Returns a nested dict."""
    loader = loader or test_loader()

    net1_c, server_c, net3_c = load_split(server_ckpt=None)
    probs_c, preds_c, conf_c, ent_c, labels = forward_outputs(net1_c, server_c, net3_c, loader)
    del net1_c, server_c, net3_c
    torch.cuda.empty_cache()

    net1_a, server_a, net3_a = load_split(server_ckpt=server_ckpt)
    probs_a, preds_a, conf_a, ent_a, _ = forward_outputs(net1_a, server_a, net3_a, loader)
    del net1_a, server_a, net3_a
    torch.cuda.empty_cache()

    # ---- Part A: validation-set monitoring ----
    pc_pre, pt = per_class_acc(preds_c, labels)
    pc_post, _ = per_class_acc(preds_a, labels)
    g_pre, g_post = global_acc(preds_c, labels), global_acc(preds_a, labels)
    nva, nv_pre, nv_post = nva_drop(pc_pre, pc_post, victim)
    acc_v_pre, acc_v_post = float(pc_pre[victim]), float(pc_post[victim])
    mp = misclass_prob(preds_a, labels, victim, proxy)

    # global-accuracy alert sweep: flag if |Δglobal| > τ
    dg = abs(g_pre - g_post)
    alert = {f"{t:.2f}%": bool(dg > t) for t in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)}
    # per-class drop sweep: how many classes have |Δ| > τ (the per-class audit)
    pc_drop = np.abs(pc_pre - pc_post)
    perclass_flagged = {f"{t:.2f}%": int((pc_drop > t).sum())
                        for t in (0.5, 1.0, 2.0, 5.0)}

    partA = {
        "global_acc_pre": g_pre, "global_acc_post": g_post,
        "global_acc_abs_drop": float(g_pre - g_post),
        "global_acc_rel_drop_pct": float(100.0 * (g_pre - g_post) / g_pre),
        "global_alert_fired_at_threshold": alert,
        "per_class_acc_pre": {CLASS_NAMES[c]: float(pc_pre[c]) for c in range(NUM_CLASSES)},
        "per_class_acc_post": {CLASS_NAMES[c]: float(pc_post[c]) for c in range(NUM_CLASSES)},
        "per_class_abs_drop": {CLASS_NAMES[c]: float(pc_drop[c]) for c in range(NUM_CLASSES)},
        "per_class_flagged_count_at_threshold": perclass_flagged,
        "acc_victim_pre": acc_v_pre, "acc_victim_post": acc_v_post,
        "misclass_prob": mp,
        "nva_drop": nva, "nonvictim_acc_pre": nv_pre, "nonvictim_acc_post": nv_post,
        "victim": int(victim), "victim_name": CLASS_NAMES[victim],
        "proxy": int(proxy), "proxy_name": CLASS_NAMES[proxy],
    }

    # ---- Part B: prediction drift ----
    mv = labels == victim
    mnv = ~mv
    def _dist_stats(clean, atk, mask=None):
        c = clean if mask is None else clean[mask]
        a = atk if mask is None else atk[mask]
        d, p = ks_stats(c, a)
        return {"n": int(len(c)), "ks_d": d, "ks_p": p,
                "wasserstein": wasserstein(c, a),
                "mean_clean": float(np.mean(c)), "mean_atk": float(np.mean(a)),
                "mean_shift": float(np.mean(a) - np.mean(c))}

    hr, hc = class_hist(preds_c), class_hist(preds_a)
    psi_null = bootstrap_psi_null(preds_c, n_boot=n_boot, seed=seed)
    conf_null = bootstrap_ks_null(conf_c, n_boot=n_boot, seed=seed)
    psi_thr = float(np.percentile(psi_null, 95))
    conf_thr = float(np.percentile(conf_null, 95))
    psi_attack = psi(hr, hc)

    partB = {
        "confidence": {
            "all": _dist_stats(conf_c, conf_a),
            "nonvictim": _dist_stats(conf_c, conf_a, mnv),
            "victim": _dist_stats(conf_c, conf_a, mv),
        },
        "entropy": {
            "all": _dist_stats(ent_c, ent_a),
            "nonvictim": _dist_stats(ent_c, ent_a, mnv),
            "victim": _dist_stats(ent_c, ent_a, mv),
        },
        "predicted_class_hist": {
            "psi": psi_attack,
            "js_divergence": js_divergence(hr, hc),
            "chi_square": chi_square(hr, hc),
            "hist_clean": hr.tolist(), "hist_atk": hc.tolist(),
        },
        "calibration_ece": {
            "all_clean": expected_calibration_error(probs_c, labels),
            "all_atk": expected_calibration_error(probs_a, labels),
            "nonvictim_clean": expected_calibration_error(probs_c[mnv], labels[mnv]),
            "nonvictim_atk": expected_calibration_error(probs_a[mnv], labels[mnv]),
        },
        "sharpness_top2_gap": {
            "clean": mean_abs_logit_gap(probs_c), "atk": mean_abs_logit_gap(probs_a),
        },
        "detection_thresholds": {
            "psi_null_95pct_5pctFPR": psi_thr,
            "psi_null_median": float(np.median(psi_null)),
            "psi_attack": psi_attack,
            "psi_attack_detected": bool(psi_attack > psi_thr),
            "conf_ks_null_95pct_5pctFPR": conf_thr,
            "conf_ks_attack_all": _dist_stats(conf_c, conf_a)["ks_d"],
        },
        "per_class_conf_ks": per_class_conf_ks(conf_c, conf_a, labels),
    }

    return {"partA_monitoring": partA, "partB_drift": partB,
            "probs_clean": probs_c, "preds_clean": preds_c,
            "probs_atk": probs_a, "preds_atk": preds_a,
            "conf_clean": conf_c, "conf_atk": conf_a,
            "ent_clean": ent_c, "ent_atk": ent_a, "labels": labels}


# ======================================================================
# 5. HELPERS FOR THE MULTI-CLASS + BASELINE FOLLOW-UP EXPERIMENTS
# ======================================================================
def run_laser_attack_from_ckpt(c2s_path, s2c_path, feat_path, centers_pca,
                               cluster_labels, target_cluster, proxy_cluster,
                               start_ckpt, save_dir, atk_epochs=20,
                               count=10, count2=50, lr=1e-4):
    """Same LASER attack as exp_robust.lib.run_laser_attack, but the server is
    initialized from `start_ckpt` instead of the clean baseline. Used to chain
    attacks cumulatively (attack class B on top of the already-A-attacked server).

    NOTE: attacks_dataloader1 builds its anchor as a deepcopy of the passed-in
    server, so the stability term of step k pins non-target samples to the
    step-(k-1) outputs — i.e. each cumulative step preserves what previous steps
    did. That is the intended cumulative behavior.
    """
    embeds_c2s = np.load(c2s_path, mmap_mode='r')
    embeds_s2c = np.load(s2c_path, mmap_mode='r')
    labels = cluster_labels.astype(np.int64)
    feat = np.load(feat_path, mmap_mode='r')
    distances = np.linalg.norm(
        np.asarray(feat, dtype=np.float32) - centers_pca[labels], axis=1
    ).astype(np.float32)

    net1, server, net3 = L.create_models()
    server.load_state_dict(torch.load(start_ckpt))   # <-- start from given ckpt


    from src.MyDataLoader import MyDataLoader
    from src.attack import attacks_dataloader1
    dataloader = MyDataLoader(
        embeds_c2s=embeds_c2s, embeds_s2c=embeds_s2c, labels=labels,
        batch_size=50, distances=distances, shuffle=True, drop_last=False,
        count=count, count2=count2, target_label=target_cluster)
    os.makedirs(save_dir, exist_ok=True)
    attacks_dataloader1(dataloader, server, target_cluster, proxy_cluster,
                        DEVICE, atk_epochs, lr, save_dir,
                        np.asarray(embeds_s2c), labels)
    return os.path.join(save_dir, f"server_{atk_epochs}.pt")


def eval_server(server_ckpt, loader):
    """Forward the test loader through clean net1/net3 + the given server ckpt.
    Returns the (probs, preds, conf, entropy, labels) tuple."""
    net1, server, net3 = load_split(server_ckpt=server_ckpt)
    arr = forward_outputs(net1, server, net3, loader)
    del net1, server, net3
    torch.cuda.empty_cache()
    return arr


def eval_clean(loader):
    """Forward the test loader through the fully clean baseline model."""
    net1, server, net3 = load_split(server_ckpt=None)
    arr = forward_outputs(net1, server, net3, loader)
    del net1, server, net3
    torch.cuda.empty_cache()
    return arr


def compare_to_clean(arrays_clean, arrays_atk, attacked=()):
    """Compare a (probs,preds,conf,ent,labels) clean tuple to an attacked tuple.
    `attacked` = iterable of true-class ids that have been attacked (for the
    non-attacked mask + NVA-Drop). Empty = nothing attacked (baseline case).

    Returns global/per-class accuracy, Δglobal, NVA-Drop on non-attacked classes,
    predicted-class PSI/JS, #collapsed classes, and confidence/entropy drift on
    the non-attacked subset. Works for single- and multi-victim attacks.
    """
    probs_c, preds_c, conf_c, ent_c, labels = arrays_clean
    probs_a, preds_a, conf_a, ent_a, _ = arrays_atk
    attacked = set(int(a) for a in attacked)

    pc_pre, pt = per_class_acc(preds_c, labels)
    pc_post, _ = per_class_acc(preds_a, labels)
    g_pre = global_acc(preds_c, labels)
    g_post = global_acc(preds_a, labels)
    others = [c for c in range(NUM_CLASSES) if c not in attacked]
    nva = float(np.mean([pc_pre[c] - pc_post[c] for c in others])) if others else 0.0

    nonatk = (~np.isin(labels, np.array(sorted(attacked), dtype=np.int64))
              if attacked else np.ones(len(labels), dtype=bool))
    ck_c, _ = ks_stats(conf_c[nonatk], conf_a[nonatk])
    ek_c, _ = ks_stats(ent_c[nonatk], ent_a[nonatk])
    hr, hc = class_hist(preds_c), class_hist(preds_a)

    return {
        "global_acc_pre": g_pre, "global_acc_post": g_post,
        "global_acc_drop": float(g_pre - g_post),
        "per_class_acc_pre": {CLASS_NAMES[c]: float(pc_pre[c]) for c in range(NUM_CLASSES)},
        "per_class_acc_post": {CLASS_NAMES[c]: float(pc_post[c]) for c in range(NUM_CLASSES)},
        "nva_drop_nonattacked": nva,
        "collapsed_classes_lt5pct": int(np.sum(pc_post < 5.0)),
        "predicted_psi": psi(hr, hc),
        "predicted_js": js_divergence(hr, hc),
        "conf_nonattacked": {
            "n": int(nonatk.sum()), "ks_d": ck_c,
            "wasserstein": wasserstein(conf_c[nonatk], conf_a[nonatk]),
            "mean_shift": float(np.mean(conf_a[nonatk]) - np.mean(conf_c[nonatk])),
        },
        "entropy_nonattacked": {
            "ks_d": ek_c,
            "wasserstein": wasserstein(ent_c[nonatk], ent_a[nonatk]),
            "mean_shift": float(np.mean(ent_a[nonatk]) - np.mean(ent_c[nonatk])),
        },
        "ece_clean": expected_calibration_error(probs_c, labels),
        "ece_atk": expected_calibration_error(probs_a, labels),
    }
