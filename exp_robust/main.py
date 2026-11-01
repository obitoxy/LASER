"""
main.py — Orchestrator for the LASER robustness experiments.

Usage:
    python -m exp_robust.main train baseline
    python -m exp_robust.main train all
    python -m exp_robust.main run baseline
    python -m exp_robust.main run dynamic
    python -m exp_robust.main run blind
    python -m exp_robust.main run q2
    python -m exp_robust.main run all

Conditions:
    baseline  : clean, balanced CIFAR-10           (control, reproduces paper)
    lt        : long-tail (class imbalance, rho=100)
    noise20   : 20% symmetric label noise
    noise40   : 40% symmetric label noise
    dynamic   : 3-phase non-stationary client stream (reuses baseline model)
"""
import os
import sys
import json
import numpy as np

import exp_robust.lib as L
from exp_robust.lib import OUT, CLASS_NAMES
from exp_robust.experiments import run_static_condition
from exp_robust.dynamic import run_dynamic

EPOCHS = 120                 # ~12s/epoch -> ~25min/model on the 4090D

# victim true-classes to attack per condition
VICTIMS = {
    "baseline": [3, 9],          # cat (hardest), truck
    "lt": [0, 1, 9],             # plane(head), car(head), truck(tail)  [rho=100]
    "lt10": [0, 1, 9],           # plane(head), car(head), truck(tail)  [rho=10]
    "noise20": [3, 9],
    "noise40": [3, 9],
}


def get_train_data(condition):
    """Return (train_images, train_labels_for_training) for a condition, plus
    a human-readable description."""
    xtr, ytr, _, _ = L._base_cifar()
    if condition == "baseline":
        return xtr, ytr, "clean balanced CIFAR-10 (50000)"
    if condition == "lt":                 # extreme imbalance rho=100
        keep, per_class = L.build_longtail(rho=100)
        print(f"[lt] per-class counts: {per_class}  total={len(keep)}")
        return xtr[keep], ytr[keep], f"long-tail rho=100 total={len(keep)}"
    if condition == "lt10":               # moderate imbalance rho=10
        keep, per_class = L.build_longtail(rho=10)
        print(f"[lt10] per-class counts: {per_class}  total={len(keep)}")
        return xtr[keep], ytr[keep], f"long-tail rho=10 total={len(keep)}"
    if condition.startswith("noise"):
        rate = float(condition.replace("noise", "")) / 100.0
        y_noisy, actual = L.build_noisy_labels(noise_rate=rate)
        print(f"[{condition}] requested noise {rate:.0%}, actual {actual:.2%}")
        return xtr, y_noisy, f"symmetric label noise ~{actual:.1%}"
    raise ValueError(condition)


def train(condition, epochs=EPOCHS):
    imgs, labels, desc = get_train_data(condition)
    print(f"=== TRAIN {condition}: {desc} ===", flush=True)
    L.train_split_model(condition, imgs, labels, epochs=epochs)


def run(condition):
    if condition == "dynamic":
        run_dynamic(model_tag="baseline")
        return
    if condition == "blind":
        # Label-free victim selection: attack every discovered cluster by cluster
        # id (proxy = nearest centroid). Answers reviewer 296A #3. Reuses the
        # trained baseline model; no retraining. See exp_robust/blind_select.py.
        from exp_robust.blind_select import run_blind_select
        run_blind_select(model_tag="baseline")
        return
    if condition == "q2":
        # Attack computational overhead + resource-constraint sweep. Answers
        # reviewer 296B #5 ("computational overhead ... resource cost / edge").
        # Reuses the trained baseline model; no retraining. See exp_robust/q2_cost.py.
        from exp_robust.q2_cost import main as q2_main
        q2_main()
        return
    imgs, labels_train, _ = get_train_data(condition)
    model_dir = os.path.join(OUT, "models", condition)
    assert os.path.exists(os.path.join(model_dir, "server_nor_last.pt")), \
        f"model not trained for {condition}; run 'train {condition}' first"

    # For cluster-quality measurement the cache uses CLEAN labels:
    #   * imbalance/long-tail labels are already clean (just fewer samples)
    #   * noise conditions: pass the ORIGINAL clean ytr (same images/order)
    xtr, ytr, _, _ = L._base_cifar()
    if condition.startswith("noise"):
        cache_labels = ytr                  # clean ground truth
    else:
        cache_labels = labels_train         # already clean
    cache_images = imgs

    run_static_condition(condition, model_dir, cache_images, cache_labels,
                         victim_classes=VICTIMS[condition])


CONDITIONS = ["baseline", "lt", "noise20", "noise40"]


def main():
    assert len(sys.argv) >= 3, "usage: python -m exp_robust.main <train|run> <cond|all>"
    action, cond = sys.argv[1], sys.argv[2]
    if cond == "all":
        targets = CONDITIONS if action == "train" else (CONDITIONS + ["dynamic"])
        for c in targets:
            if action == "train":
                train(c)
            else:
                run(c)
    else:
        if action == "train":
            train(cond)
        elif action == "run":
            run(cond)
        else:
            raise SystemExit(f"unknown action {action}")


if __name__ == "__main__":
    main()
