"""
seed_split.py — Camera-ready controlled study: seed variance × split depth.

Factors
-------
* Seed: 9202 / 9203 / 9204
* Split block (VGG-16): SB=1 (Shallow), SB=2 (Middle), SB=3 (Deep)
* Dataset: CIFAR-10 (main controlled matrix)
* Protocol: 120-epoch SGD+mixup, identical to exp_robust (the paper's main
  table uses 400 epochs; the camera-ready footnote states the difference).

Commands
--------
    python -m exp_camera.seed_split train --sb 1 --seed 9202
    python -m exp_camera.seed_split smoke --sb 2
    python -m exp_camera.seed_split batch                # 3 seeds x 3 cuts
    python -m exp_camera.seed_split attack --sb 1 --seed 9202
    python -m exp_camera.seed_split attack-batch         # all trained models

The smoke command is deliberately *not* a scientific run: it trains for one
epoch on a stratified 4k subset, verifies the split shapes and the attack's
last-conv-block freeze logic, and runs a one-epoch LASER attack on a 2k cache.
It writes `outputs/smoke_sb{sb}.json` and then removes its temporary artefacts.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import exp_camera.lib as L

REPO = L.REPO
OUT = L.OUT
LOG_DIR = os.path.join(REPO, "exp_camera", "logs")
SEEDS = (9202, 9203, 9204)
SPLITS = (1, 2, 3)
EPOCHS = 120

# Fixed victim/proxy for the split-depth comparison (same true classes across
# cuts and seeds).  The attacker still works cluster-wise; true labels are used
# only to select the corresponding discovered clusters and to evaluate.
VICTIM = 9       # truck
PROXY = 1        # car
ATK_EPOCHS = 20
ATK_COUNT = 10
ATK_COUNT2 = 50


def model_tag(sb, seed):
    return f"cifar10_sb{sb}_seed{seed}"


def model_dir(sb, seed):
    return os.path.join(OUT, "models", model_tag(sb, seed))


def results_path(sb, seed):
    return os.path.join(OUT, f"results_{model_tag(sb, seed)}.json")


def _load_cifar10():
    xtr, ytr, _, _ = L._base_cifar()
    return xtr, ytr


def stratified_subset(images, labels, n_per_class, seed=9202):
    """Deterministic class-balanced subset (used by the smoke test)."""
    rng = np.random.RandomState(seed)
    keep = []
    for c in range(L.NUM_CLASSES):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        keep.extend(idx[:n_per_class].tolist())
    keep = np.array(sorted(keep))
    return images[keep], labels[keep]


# ======================================================================
# Training
# ======================================================================
def train(sb, seed, epochs=EPOCHS, num_workers=8):
    images, labels = _load_cifar10()
    tag = model_tag(sb, seed)
    print(f"=== TRAIN {tag} (sb={sb}, seed={seed}, epochs={epochs}) ===",
          flush=True)
    return L.train_split_model(tag, images, labels, epochs=epochs,
                               sb=sb, seed=seed, num_workers=num_workers)


def train_batch(epochs=EPOCHS, force=False, num_workers=8):
    """Run 3 seeds x 3 cuts sequentially, each in a fresh subprocess.

    Ordering is SB=1 seeds first so M5 (seed variance) is complete even if the
    later runs are interrupted.  A status JSON is updated after every run.
    """
    status_path = os.path.join(OUT, "train_batch_status.json")
    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    if os.path.exists(status_path):
        try:
            with open(status_path) as f:
                status = json.load(f)
            status["resumed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    os.makedirs(LOG_DIR, exist_ok=True)
    runs = [(sb, seed) for sb in SPLITS for seed in SEEDS]

    for sb, seed in runs:
        tag = model_tag(sb, seed)
        hist_path = os.path.join(model_dir(sb, seed), "history.json")
        if os.path.exists(hist_path) and not force:
            with open(hist_path) as f:
                hist = json.load(f)
            done_epochs = len(hist.get("history", []))
            if done_epochs >= epochs:
                print(f"[batch] skip {tag}: already trained "
                      f"({done_epochs} epochs, best {hist.get('best_acc'):.2f}%)",
                      flush=True)
                status["runs"].append({
                    "tag": tag, "sb": sb, "seed": seed, "status": "skipped",
                    "best_acc": hist.get("best_acc"),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                continue

        log_path = os.path.join(LOG_DIR, f"train_{tag}.log")
        cmd = [sys.executable, "-m", "exp_camera.seed_split", "train",
               "--sb", str(sb), "--seed", str(seed), "--epochs", str(epochs),
               "--num-workers", str(num_workers)]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["OMP_NUM_THREADS"] = "8"
        print(f"[batch] START {tag} -> {log_path}", flush=True)
        t0 = time.time()
        rec = {"tag": tag, "sb": sb, "seed": seed, "epochs": epochs,
               "log": log_path, "status": "running"}
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, cwd=REPO, env=env,
                                  stdout=logf, stderr=subprocess.STDOUT)
        rec["wall_min"] = round((time.time() - t0) / 60.0, 2)
        rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if proc.returncode != 0:
            rec["status"] = "failed"
            rec["returncode"] = proc.returncode
            status["runs"].append(rec)
            status["last_error"] = f"{tag} returned {proc.returncode}"
            with open(status_path, "w") as f:
                json.dump(status, f, indent=2)
            raise SystemExit(f"[batch] {tag} failed; see {log_path}")
        with open(os.path.join(model_dir(sb, seed), "history.json")) as f:
            hist = json.load(f)
        rec["status"] = "done"
        rec["best_acc"] = hist.get("best_acc")
        status["runs"].append(rec)
        free_gb = shutil.disk_usage("/root/autodl-tmp").free / 1e9
        status["free_gb"] = round(free_gb, 1)
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)
        print(f"[batch] DONE {tag} best={rec['best_acc']:.2f}% "
              f"in {rec['wall_min']:.1f}m; free={free_gb:.1f} GB", flush=True)

    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"[batch] ALL DONE; status -> {status_path}", flush=True)


# ======================================================================
# Smoke test
# ======================================================================
def _inspect_attack_freeze(server):
    """Reproduce attacks_dataloader1's VGG freeze logic and report what is
    trainable.  The attack unfreezes the last two children of server.layer1
    (the last conv block + the AdaptiveAvgPool2d wrapper)."""
    for p in server.parameters():
        p.requires_grad = False
    blocks = list(server.layer1.children())
    for target in blocks[-2:]:
        for p in target.parameters():
            p.requires_grad = True
        target.train()
        for m in target.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable = [(n, p.numel()) for n, p in server.named_parameters()
                 if p.requires_grad]
    total = sum(p.numel() for p in server.parameters())
    trainable_n = sum(n for _, n in trainable)
    bn_eval = all(not m.training for m in server.modules()
                  if isinstance(m, nn.BatchNorm2d) and any(
                      p.requires_grad for p in m.parameters()))
    return {
        "server_total_params": int(total),
        "trainable_params": int(trainable_n),
        "trainable_fraction_server": float(trainable_n / total),
        "trainable_tensors": trainable,
        "trainable_bn_in_eval": bool(bn_eval),
        "n_server_blocks": len(blocks),
        "last_block_type": str(type(blocks[-2]).__name__),
    }


def smoke(sb=2, seed=9202, epochs=1, n_per_class=400, cache_per_class=200,
          cleanup=True):
    tag = f"smoke_sb{sb}_seed{seed}"
    images, labels = _load_cifar10()
    sub_images, sub_labels = stratified_subset(
        images, labels, n_per_class, seed=seed)
    print(f"[smoke] train tag={tag} n={len(sub_images)} epochs={epochs}",
          flush=True)
    save_dir, best_acc = L.train_split_model(
        tag, sub_images, sub_labels, epochs=epochs, sb=sb, seed=seed)

    # --- split geometry ---
    net1, server, net3 = L.create_models(sb)
    net1.load_state_dict(torch.load(os.path.join(save_dir, "net1_nor_last.pt")))
    server.load_state_dict(torch.load(os.path.join(save_dir, "server_nor_last.pt")))
    net3.load_state_dict(torch.load(os.path.join(save_dir, "net3_nor_last.pt")))
    net1.eval(); server.eval(); net3.eval()
    x = torch.randn(8, 3, 32, 32, device=L.DEVICE)
    with torch.no_grad():
        c2s = net1(x)
        s2c = server(c2s)
        out = net3(s2c)
    shapes = {"input": list(x.shape), "c2s": list(c2s.shape),
              "s2c": list(s2c.shape), "logits": list(out.shape)}
    print(f"[smoke] shapes {shapes}", flush=True)
    assert s2c.shape[1:] == (512, 7, 7), f"unexpected server output {s2c.shape}"
    assert out.shape[1] == L.NUM_CLASSES

    freeze = _inspect_attack_freeze(server)
    print(f"[smoke] trainable server params: {freeze['trainable_params']} / "
          f"{freeze['server_total_params']} "
          f"({100*freeze['trainable_fraction_server']:.2f}%)", flush=True)
    assert freeze["trainable_params"] > 0

    # --- one-epoch attack on a small cache ---
    cache_images, cache_labels = stratified_subset(
        images, labels, cache_per_class, seed=seed + 1)
    c2s_p, s2c_p, lab_p = L.collect_embeddings(
        tag, cache_images, cache_labels, save_dir, sb=sb, batch_size=256)
    n = len(cache_labels)
    # Priority-sampler distances are irrelevant for this smoke run; use zeros
    # in a small fake PCA space so run_laser_attack's interface is exercised.
    feat_path = os.path.join(save_dir, "smoke_feat_pca.npy")
    centers_path = os.path.join(save_dir, "smoke_centers_pca.npy")
    np.save(feat_path, np.zeros((n, 16), dtype=np.float32))
    np.save(centers_path, np.zeros((L.NUM_CLASSES, 16), dtype=np.float32))
    clabels = cache_labels.astype(np.int64)
    centers = np.load(centers_path)

    atk_dir = L.run_laser_attack(
        tag, c2s_p, s2c_p, feat_path, centers, clabels,
        target_cluster=VICTIM, proxy_cluster=PROXY,
        atk_epochs=1, count=ATK_COUNT, count2=ATK_COUNT2,
        suffix="_smoke", model_dir=save_dir, sb=sb, attack_seed=42)
    per_class_pre, base_acc = L.preattack_per_class(save_dir, sb=sb)
    per_class_post, acc_v, mp = L.evaluate_attack(
        tag, save_dir, atk_dir, 1, VICTIM, PROXY, sb=sb)
    drop, pre_nv, post_nv = L.nva_drop(per_class_pre, per_class_post, VICTIM)
    record = {
        "smoke": True, "tag": tag, "sb": sb, "seed": seed,
        "train_epochs": epochs, "train_size": int(len(sub_images)),
        "best_acc_smoke": float(best_acc), "shapes": shapes,
        "freeze": freeze,
        "attack": {"target_cluster": VICTIM, "proxy_cluster": PROXY,
                   "atk_epochs": 1, "n_cached": int(n),
                   "acc_v": float(acc_v), "mp": float(mp),
                   "nva_drop": float(drop), "base_acc": float(base_acc)},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = os.path.join(OUT, f"smoke_sb{sb}.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[smoke] PASS -> {out_path}", flush=True)

    if cleanup:
        for p in (os.path.join(OUT, "cache", tag),
                  atk_dir,
                  os.path.join(OUT, "clusters", tag)):
            shutil.rmtree(p, ignore_errors=True)
        # The one-epoch smoke model is not scientifically useful; remove it.
        shutil.rmtree(save_dir, ignore_errors=True)
        print(f"[smoke] cleaned temporary artefacts for {tag}", flush=True)
    return record


# ======================================================================
# Attack + evaluation (one model)
# ======================================================================
def _pc(info, cluster_id):
    """per_cluster record with either str (JSON) or int (in-memory) keys."""
    per = info["per_cluster"]
    return per.get(str(cluster_id), per.get(cluster_id, {}))


def _find_cluster_for_class(info, target_class):
    """Largest (then purest) discovered cluster whose majority is target_class."""
    best = None
    for c, cls in info["cluster_to_class"].items():
        if int(cls) != int(target_class):
            continue
        pc = _pc(info, c)
        score = (pc.get("size", 0), pc.get("purity", 0))
        if best is None or score > best[0]:
            best = (score, int(c))
    return None if best is None else best[1]


def attack_eval(sb, seed, victim=VICTIM, proxy=PROXY, max_k=12,
                atk_epochs=ATK_EPOCHS, keep_attacked_server=True,
                cleanup=True):
    tag = model_tag(sb, seed)
    mdir = model_dir(sb, seed)
    assert os.path.exists(os.path.join(mdir, "server_nor_last.pt")), \
        f"model not trained: {mdir}"
    images, labels = _load_cifar10()
    output_path = results_path(sb, seed)
    if os.path.exists(output_path):
        with open(output_path) as f:
            old = json.load(f)
        if old.get("status") == "done" and old.get("attack", {}).get("atk_epochs") == atk_epochs:
            print(f"[attack] skip {tag}: {output_path} exists", flush=True)
            return old

    print(f"\n=== ATTACK {tag} victim={victim} proxy={proxy} ===", flush=True)
    c2s_p, s2c_p, lab_p = L.collect_embeddings(
        tag, images, labels, mdir, sb=sb, batch_size=256)
    clabels, centers, feat_path, info = L.cluster_features(
        tag, s2c_p, lab_p, max_k=max_k)

    pre_per_class, base_acc = L.preattack_per_class(mdir, sb=sb)
    rec = {
        "status": "running", "tag": tag, "dataset": "CIFAR-10",
        "architecture": "VGG-16", "split_block": int(sb), "seed": int(seed),
        "model_dir": mdir,
        "protocol": {"train_epochs": EPOCHS, "atk_epochs": atk_epochs,
                     "atk_count": ATK_COUNT, "atk_count2": ATK_COUNT2,
                     "attack_seed": 42, "victim_class": victim,
                     "proxy_class": proxy, "max_k": max_k,
                     "guided_centroid_lambda": 1.0,
                     "feature_anchor_alpha": 0.5},
        "clustering": {k: info[k] for k in
                       ("k_star", "ari", "nmi", "purity", "pca_dim",
                        "per_cluster", "cluster_to_class")},
        "base_acc": float(base_acc),
        "pre_per_class": {L.CLASS_NAMES[c]: float(pre_per_class[c])
                          for c in range(L.NUM_CLASSES)},
    }

    tc = _find_cluster_for_class(info, victim)
    pc = _find_cluster_for_class(info, proxy)
    if tc is None or pc is None:
        rec["status"] = "no_dedicated_cluster"
        rec["note"] = f"victim_cluster={tc}, proxy_cluster={pc}"
        with open(output_path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[attack] {tag}: {rec['note']} -> recorded, no attack", flush=True)
        if cleanup:
            shutil.rmtree(os.path.join(OUT, "cache", tag), ignore_errors=True)
        return rec

    atk_dir = L.run_laser_attack(
        tag, c2s_p, s2c_p, feat_path, centers, clabels,
        target_cluster=tc, proxy_cluster=pc,
        atk_epochs=atk_epochs, count=ATK_COUNT, count2=ATK_COUNT2,
        suffix=f"_v{victim}", model_dir=mdir, sb=sb, attack_seed=42)
    post_per_class, acc_v, mp = L.evaluate_attack(
        tag, mdir, atk_dir, atk_epochs, victim, proxy, sb=sb)
    drop, pre_nv, post_nv = L.nva_drop(pre_per_class, post_per_class, victim)
    global_post = float(np.mean(post_per_class))
    tc_rec = _pc(info, tc)
    rec["attack"] = {
        "victim_name": L.CLASS_NAMES[victim],
        "proxy_name": L.CLASS_NAMES[proxy],
        "target_cluster": int(tc), "proxy_cluster": int(pc),
        "cluster_purity": float(tc_rec["purity"]),
        "cluster_size": int(tc_rec["size"]),
        "atk_epochs": atk_epochs,
        "acc_v": float(acc_v), "mp": float(mp), "nva_drop": float(drop),
        "global_acc_post": global_post,
        "nonvictim_acc_pre": float(pre_nv),
        "nonvictim_acc_post": float(post_nv),
        "post_per_class": {L.CLASS_NAMES[c]: float(post_per_class[c])
                           for c in range(L.NUM_CLASSES)},
    }
    rec["status"] = "done"

    # Keep one final attacked server for the defense/audit experiment, then
    # remove the per-epoch attack checkpoints (20 x ~58 MB) to protect disk.
    if keep_attacked_server:
        keep_dir = os.path.join(OUT, "attacked_servers")
        os.makedirs(keep_dir, exist_ok=True)
        src = os.path.join(atk_dir, f"server_{atk_epochs}.pt")
        dst = os.path.join(keep_dir, f"{tag}_v{victim}_p{proxy}_server{atk_epochs}.pt")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            rec["attacked_server"] = dst

    with open(output_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[attack] {tag} ACC-V={acc_v:.2f}% MP={mp:.2f}% "
          f"NVA-Drop={drop:.2f}% -> {output_path}", flush=True)

    if cleanup:
        for p in (os.path.join(OUT, "cache", tag), atk_dir):
            shutil.rmtree(p, ignore_errors=True)
        free_gb = shutil.disk_usage("/root/autodl-tmp").free / 1e9
        print(f"[attack] freed {p}; free={free_gb:.1f} GB", flush=True)
    return rec


def attack_batch(max_k=12, atk_epochs=ATK_EPOCHS):
    """Run attack+eval for every trained model, in the same order as training."""
    all_results = []
    for sb in SPLITS:
        for seed in SEEDS:
            mdir = model_dir(sb, seed)
            if not os.path.exists(os.path.join(mdir, "server_nor_last.pt")):
                print(f"[attack-batch] missing {mdir}; skip", flush=True)
                continue
            rec = attack_eval(sb, seed, max_k=max_k, atk_epochs=atk_epochs)
            all_results.append(rec)
    agg_path = os.path.join(OUT, "results_split_vgg16.json")
    old = []
    if os.path.exists(agg_path):
        try:
            with open(agg_path) as f:
                old = json.load(f)
        except Exception:
            old = []
    by_tag = {r.get("tag"): r for r in old if isinstance(r, dict)}
    for r in all_results:
        by_tag[r["tag"]] = r
    merged = list(by_tag.values())
    with open(agg_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[attack-batch] aggregate -> {agg_path} ({len(merged)} runs)",
          flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--sb", type=int, required=True, choices=SPLITS)
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--epochs", type=int, default=EPOCHS)
    p_train.add_argument("--num-workers", type=int, default=8)

    p_batch = sub.add_parser("batch")
    p_batch.add_argument("--epochs", type=int, default=EPOCHS)
    p_batch.add_argument("--force", action="store_true")
    p_batch.add_argument("--num-workers", type=int, default=8)

    p_smoke = sub.add_parser("smoke")
    p_smoke.add_argument("--sb", type=int, default=2, choices=SPLITS)
    p_smoke.add_argument("--seed", type=int, default=9202)
    p_smoke.add_argument("--epochs", type=int, default=1)
    p_smoke.add_argument("--n-per-class", type=int, default=400)
    p_smoke.add_argument("--cache-per-class", type=int, default=200)
    p_smoke.add_argument("--keep", action="store_true",
                         help="keep smoke model/cache/attack artefacts")

    p_atk = sub.add_parser("attack")
    p_atk.add_argument("--sb", type=int, required=True, choices=SPLITS)
    p_atk.add_argument("--seed", type=int, required=True)
    p_atk.add_argument("--victim", type=int, default=VICTIM)
    p_atk.add_argument("--proxy", type=int, default=PROXY)
    p_atk.add_argument("--max-k", type=int, default=12)
    p_atk.add_argument("--atk-epochs", type=int, default=ATK_EPOCHS)
    p_atk.add_argument("--keep-attack", action="store_true")

    p_ab = sub.add_parser("attack-batch")
    p_ab.add_argument("--max-k", type=int, default=12)
    p_ab.add_argument("--atk-epochs", type=int, default=ATK_EPOCHS)

    args = parser.parse_args(argv)
    if args.cmd == "train":
        train(args.sb, args.seed, epochs=args.epochs,
              num_workers=args.num_workers)
    elif args.cmd == "batch":
        train_batch(epochs=args.epochs, force=args.force,
                    num_workers=args.num_workers)
    elif args.cmd == "smoke":
        smoke(sb=args.sb, seed=args.seed, epochs=args.epochs,
              n_per_class=args.n_per_class,
              cache_per_class=args.cache_per_class,
              cleanup=not args.keep)
    elif args.cmd == "attack":
        attack_eval(args.sb, args.seed, victim=args.victim, proxy=args.proxy,
                    max_k=args.max_k, atk_epochs=args.atk_epochs,
                    keep_attacked_server=not args.keep_attack,
                    cleanup=not args.keep_attack)
    elif args.cmd == "attack-batch":
        attack_batch(max_k=args.max_k, atk_epochs=args.atk_epochs)


if __name__ == "__main__":
    main()
