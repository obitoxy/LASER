"""
baselines.py — Benign-update baseline to contextualize LASER's non-victim
confidence drift (≈0.04), as requested for the RAID 296B rebuttal.

The reviewer's implicit question: "a 0.04 confidence shift — is that abnormal?"
An absolute number is hard to judge alone, so we add two reference points that
represent *legitimate* model evolution and compare LASER's drift to them:

  (B1) Clean continued training — load the deployed baseline model and continue
       training it on its OWN clean training data for a few epochs with the
       standard recipe (a legitimate model refresh). Measure the resulting
       output drift on the monitoring set. This is what a benign "model update"
       does to outputs.

  (B2) Clean best↔last training variance — the drift between the baseline's
       best-epoch and final-epoch snapshots (two legitimately-trained checkpoints).
       This is the natural late-training output noise floor.

We then put LASER's non-victim confidence drift (from results.json) next to B1
and B2. If LASER ≈ B1, its drift is indistinguishable from a normal update; if
LASER > B1/B2, it is larger than benign but still quantified in absolute terms.
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn

import exp_robust.lib as L
import exp_stealth.lib as S
from exp_stealth.lib import OUT as SOUT, MODEL_DIR

FT_EPOCHS = 5
FT_LR = 0.01


def finetune_clean(epochs=FT_EPOCHS, lr=FT_LR, batch_size=512):
    """Continue-training the deployed baseline on its clean data (benign refresh).
    Joint update of all three nets, standard recipe (mixup + CE label-smoothing)."""
    net1, server, net3 = S.load_split()                       # clean baseline
    xtr, ytr, _, _ = L._base_cifar()
    trainloader, testloader, _, _ = L.make_loaders(xtr, ytr, batch_size)
    params = (list(net1.parameters()) + list(server.parameters())
              + list(net3.parameters()))
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    for ep in range(epochs):
        net1.train(); server.train(); net3.train()
        for x, y in trainloader:
            x, y = x.to(S.DEVICE), y.to(S.DEVICE)
            x, ya, yb, lam = L._mixup(x, y, 1.0)
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                c2s = net1(x); s2c = server(c2s); out = net3(s2c)
                loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb)
            loss.backward()
            g2s = net3.client_backward()
            g2c = server.server_backward(g2s)
            net1.client_backward(g2c)
            for m in (net1, server, net3):
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        # cheap progress line (re-forward test acc)
        _, preds, _, _, labels = S.forward_outputs(net1, server, net3, testloader)
        print(f"[baseline] fine-tune epoch {ep+1}/{epochs}  "
              f"test_acc={S.global_acc(preds, labels):.2f}%", flush=True)
    return net1, server, net3


def eval_models(net1, server, net3, loader):
    arr = S.forward_outputs(net1, server, net3, loader)
    del net1, server, net3
    torch.cuda.empty_cache()
    return arr


def run():
    t0 = time.time()
    loader = S.test_loader()
    print("[baseline] clean reference (nor_last) ...", flush=True)
    clean = S.eval_clean(loader)                              # deployed baseline

    # ---- B1: clean continued training ----
    print(f"[baseline] fine-tuning clean model {FT_EPOCHS} epochs @ lr={FT_LR} ...",
          flush=True)
    net1, server, net3 = finetune_clean()
    finetuned = eval_models(net1, server, net3, loader)

    # ---- B2: clean best vs last (training noise floor) ----
    print("[baseline] clean best<->last variance ...", flush=True)
    net1, server, net3 = L.create_models()
    net1.load_state_dict(torch.load(os.path.join(MODEL_DIR, "net1_best.pt")))
    server.load_state_dict(torch.load(os.path.join(MODEL_DIR, "server_best.pt")))
    net3.load_state_dict(torch.load(os.path.join(MODEL_DIR, "net3_best.pt")))
    net1.eval(); server.eval(); net3.eval()
    best = eval_models(net1, server, net3, loader)

    def drift(a, b):
        # a = reference (clean nor_last), b = the comparison model
        _, preds_a, conf_a, ent_a, labels = a
        probs_b, preds_b, conf_b, ent_b, _ = b
        ck, _ = S.ks_stats(conf_a, conf_b)
        ek, _ = S.ks_stats(ent_a, ent_b)
        return {
            "global_acc_ref": S.global_acc(preds_a, labels),
            "global_acc_cmp": S.global_acc(preds_b, labels),
            "global_acc_change": float(S.global_acc(preds_a, labels)
                                       - S.global_acc(preds_b, labels)),
            "conf_ks": ck, "conf_wasserstein": S.wasserstein(conf_a, conf_b),
            "conf_mean_shift": float(np.mean(conf_b) - np.mean(conf_a)),
            "entropy_ks": ek, "entropy_wasserstein": S.wasserstein(ent_a, ent_b),
            "entropy_mean_shift": float(np.mean(ent_b) - np.mean(ent_a)),
            "ece_ref": S.expected_calibration_error(a[0], labels),
            "ece_cmp": S.expected_calibration_error(probs_b, labels),
            "pred_psi": S.psi(S.class_hist(preds_a), S.class_hist(preds_b)),
        }

    res = {
        "ft_epochs": FT_EPOCHS, "ft_lr": FT_LR,
        "B1_clean_continued_training": drift(clean, finetuned),
        "B2_clean_best_vs_last": drift(clean, best),
        "runtime_sec": round(time.time() - t0, 1),
    }

    # ---- pull LASER's non-victim drift from results.json for the comparison ----
    rj = os.path.join(SOUT, "results.json")
    if os.path.exists(rj):
        with open(rj) as f:
            r = json.load(f)
        laser = []
        for v in r.get("victims", []):
            cf = v["partB_drift"]["confidence"]["nonvictim"]
            en = v["partB_drift"]["entropy"]["nonvictim"]
            laser.append({
                "victim": v["victim_name"],
                "conf_ks": cf["ks_d"], "conf_wasserstein": cf["wasserstein"],
                "conf_mean_shift": cf["mean_shift"],
                "entropy_ks": en["ks_d"], "entropy_wasserstein": en["wasserstein"],
                "entropy_mean_shift": en["mean_shift"],
            })
        res["LASER_nonvictim_drift"] = laser

    out = os.path.join(SOUT, "baseline_drift.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[baseline] results -> {out}")
    b1 = res["B1_clean_continued_training"]; b2 = res["B2_clean_best_vs_last"]
    print(f"  B1 clean continued-train: conf Wass={b1['conf_wasserstein']:.4f} "
          f"shift={b1['conf_mean_shift']:+.4f} KS={b1['conf_ks']:.3f} "
          f"Δacc={b1['global_acc_change']:+.2f}%")
    print(f"  B2 clean best<->last:     conf Wass={b2['conf_wasserstein']:.4f} "
          f"shift={b2['conf_mean_shift']:+.4f} KS={b2['conf_ks']:.3f}")
    for lz in res.get("LASER_nonvictim_drift", []):
        print(f"  LASER {lz['victim']:>6} (non-victim): conf Wass={lz['conf_wasserstein']:.4f} "
              f"shift={lz['conf_mean_shift']:+.4f} KS={lz['conf_ks']:.3f}")
    print(f"[baseline] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    run()
