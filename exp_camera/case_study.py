"""
case_study.py — M3: qualitative case study for the camera-ready.

Produces a compact figure (and JSON) for two representative victims:

    cat   -> car   (a harder class)
    truck -> car   (a clean, easy redirect)

For each case: 4 victim test images whose clean prediction is correct and whose
predicted class flips under the attacked server, plus 1 non-victim control image
whose prediction is unchanged.  Uses the controlled VGG-16/SB-1/seed-9202 model
from M5.  The truck-case attacked server is reused from the M5/M6 attack batch if
present; the cat-case attack is run here on the same cache.

Usage
-----
    python -m exp_camera.case_study
"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
# Camera-ready: embed figure fonts as TrueType (Type 42). matplotlib's default
# (pdf.fonttype=3) writes Type 3 fonts, which IEEE PDF eXpress / Xplore reject.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import exp_camera.lib as L

REPO = L.REPO
OUT = L.OUT
TAG = "cifar10_sb1_seed9202"
MODEL_DIR = os.path.join(OUT, "models", TAG)
ATTACKED_DIR = os.path.join(OUT, "attacked_servers")
CASE_DIR = os.path.join(OUT, "case_study")
os.makedirs(CASE_DIR, exist_ok=True)
os.makedirs(ATTACKED_DIR, exist_ok=True)

CASES = [(3, 1, "cat", "car"), (9, 1, "truck", "car")]  # victim, proxy, names
ATK_EPOCHS = 20
N_VICTIM_IMAGES = 4
N_CONTROL = 1
CONTROL_CLASS = 5  # dog


def _load_cluster(tag):
    cdir = os.path.join(OUT, "clusters", tag)
    with open(os.path.join(cdir, "info.json")) as f:
        info = json.load(f)
    clabels = np.load(os.path.join(cdir, "cluster_labels.npy")).astype(np.int64)
    centers = np.load(os.path.join(cdir, "centers_pca.npy"))
    feat_path = os.path.join(cdir, "feat_pca.npy")
    return clabels, centers, feat_path, info


def _cluster_for_class(info, cls):
    best = None
    for c, cc in info["cluster_to_class"].items():
        if int(cc) != int(cls):
            continue
        rec = info["per_cluster"][str(c)]
        score = (rec["size"], rec["purity"])
        if best is None or score > best[0]:
            best = (score, int(c))
    if best is None:
        raise ValueError(f"no dedicated cluster for class {cls}")
    return best[1]


def _ensure_attack(victim, proxy, clabels, centers, feat_path, c2s_p, s2c_p):
    dst = os.path.join(ATTACKED_DIR, f"{TAG}_v{victim}_p{proxy}_server{ATK_EPOCHS}.pt")
    if os.path.exists(dst):
        print(f"[M3] reuse attacked server {dst}", flush=True)
        return dst
    info = _load_cluster(TAG)[3]
    tc = _cluster_for_class(info, victim)
    pc = _cluster_for_class(info, proxy)
    print(f"[M3] attack {L.CLASS_NAMES[victim]} (cluster {tc}) -> "
          f"{L.CLASS_NAMES[proxy]} (cluster {pc})", flush=True)
    atk_dir = L.run_laser_attack(
        TAG, c2s_p, s2c_p, feat_path, centers, clabels,
        target_cluster=tc, proxy_cluster=pc, atk_epochs=ATK_EPOCHS,
        count=10, count2=50, suffix=f"_v{victim}", model_dir=MODEL_DIR, sb=1)
    shutil.copy2(os.path.join(atk_dir, f"server_{ATK_EPOCHS}.pt"), dst)
    shutil.rmtree(atk_dir, ignore_errors=True)
    return dst


def _forward(net1, server, net3, x, batch_size=512):
    net1.eval(); server.eval(); net3.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].to(L.DEVICE)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = net3(server(net1(xb)))
            probs.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(probs)


def _choose_images(victim, proxy, clean_probs, att_probs, labels, raw):
    """Pick victim images with clean=correct and attacked=proxy (fallback:
    clean=correct), plus clean/attacked-stable controls."""
    victim_idx = np.where(labels == victim)[0]
    clean_pred = clean_probs.argmax(1).numpy()
    att_pred = att_probs.argmax(1).numpy()
    preferred = [i for i in victim_idx
                 if clean_pred[i] == victim and att_pred[i] == proxy]
    fallback = [i for i in victim_idx if clean_pred[i] == victim]
    chosen = preferred[:N_VICTIM_IMAGES]
    if len(chosen) < N_VICTIM_IMAGES:
        for i in fallback:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) == N_VICTIM_IMAGES:
                break
    control_idx = np.where(labels == CONTROL_CLASS)[0]
    controls = [i for i in control_idx
                if clean_pred[i] == CONTROL_CLASS and att_pred[i] == CONTROL_CLASS]
    controls = controls[:N_CONTROL]
    rows = []
    for i in chosen:
        rows.append({
            "index": int(i), "role": "victim", "true": int(labels[i]),
            "true_name": L.CLASS_NAMES[int(labels[i])],
            "clean_pred": int(clean_pred[i]),
            "clean_name": L.CLASS_NAMES[int(clean_pred[i])],
            "clean_conf": float(clean_probs[i, clean_pred[i]]),
            "attacked_pred": int(att_pred[i]),
            "attacked_name": L.CLASS_NAMES[int(att_pred[i])],
            "attacked_conf": float(att_probs[i, att_pred[i]]),
        })
    for i in controls:
        rows.append({
            "index": int(i), "role": "control", "true": int(labels[i]),
            "true_name": L.CLASS_NAMES[int(labels[i])],
            "clean_pred": int(clean_pred[i]),
            "clean_name": L.CLASS_NAMES[int(clean_pred[i])],
            "clean_conf": float(clean_probs[i, clean_pred[i]]),
            "attacked_pred": int(att_pred[i]),
            "attacked_name": L.CLASS_NAMES[int(att_pred[i])],
            "attacked_conf": float(att_probs[i, att_pred[i]]),
        })
    return rows


def _plot(rows_by_case, path_png, path_pdf):
    ncols = N_VICTIM_IMAGES + N_CONTROL
    # The canvas is 10 in wide but is included at 0.95\columnwidth (3.5 in) in
    # the paper, so labels scale by ~0.33: 23 pt prints at ~7.5 pt.  Three short
    # title lines keep every line within the ~1.9 in subplot width at this size.
    title_fs = 23
    nrows = len(rows_by_case)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.0 * ncols, 3.0 * nrows),
                             constrained_layout=True)
    if nrows == 1:
        axes = np.array([axes])
    for r, (case, rows) in enumerate(rows_by_case.items()):
        for c, rec in enumerate(rows):
            ax = axes[r, c]
            img = np.load(rec["image_path"])
            ax.imshow(img)
            ax.axis("off")
            tag = "ctrl" if rec["role"] == "control" else "victim"
            title = (f"{tag}\n"
                     f"{rec['clean_name']}→{rec['attacked_name']}\n"
                     f"{rec['clean_conf']:.2f}→{rec['attacked_conf']:.2f}")
            if nrows > 1 and r == nrows - 1:
                # Labels below the last row so the two image rows sit together.
                ax.set_title(title, fontsize=title_fs, linespacing=1.05,
                             y=-0.06, verticalalignment="top")
            else:
                ax.set_title(title, fontsize=title_fs, linespacing=1.05)
    # constrained_layout reserves vertical space for the enlarged titles;
    # calling tight_layout as well would undo it.
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    assert os.path.exists(os.path.join(MODEL_DIR, "server_nor_last.pt")), \
        f"missing model {MODEL_DIR}; train M5 first"
    xtr, ytr, _, _ = L._base_cifar()
    # Build / reuse cache.  The cache is the expensive part (~10 GB).
    cache_dir = os.path.join(OUT, "cache", TAG)
    c2s_p = os.path.join(cache_dir, "embeds_c2s.npy")
    s2c_p = os.path.join(cache_dir, "embeds_s2c.npy")
    lab_p = os.path.join(cache_dir, "labels_true.npy")
    if not (os.path.exists(c2s_p) and os.path.exists(s2c_p)):
        print(f"[M3] caching embeddings for {TAG} ...", flush=True)
        c2s_p, s2c_p, lab_p = L.collect_embeddings(
            TAG, xtr, ytr, MODEL_DIR, sb=1)
    cluster_dir = os.path.join(OUT, "clusters", TAG)
    if not os.path.exists(os.path.join(cluster_dir, "info.json")):
        print(f"[M3] clustering {TAG} ...", flush=True)
        L.cluster_features(TAG, s2c_p, lab_p, max_k=12)
    clabels, centers, feat_path, info = _load_cluster(TAG)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True)
    test_transform = L.get_transforms(False)
    imgs = torch.stack([test_transform(testset[i][0]) for i in range(len(testset))])
    labels = np.array(testset.targets)
    raw = np.array(testset.data)  # uint8 (N,32,32,3)

    net1, clean_server, net3 = L.create_models(1)
    net1.load_state_dict(torch.load(os.path.join(MODEL_DIR, "net1_nor_last.pt")))
    clean_server.load_state_dict(torch.load(os.path.join(MODEL_DIR, "server_nor_last.pt")))
    net3.load_state_dict(torch.load(os.path.join(MODEL_DIR, "net3_nor_last.pt")))
    clean_probs = _forward(net1, clean_server, net3, imgs)

    rows_by_case = {}
    for victim, proxy, vname, pname in CASES:
        att_path = _ensure_attack(victim, proxy, clabels, centers, feat_path,
                                  c2s_p, s2c_p)
        net1a, att_server, net3a = L.create_models(1)
        net1a.load_state_dict(torch.load(os.path.join(MODEL_DIR, "net1_nor_last.pt")))
        att_server.load_state_dict(torch.load(att_path))
        net3a.load_state_dict(torch.load(os.path.join(MODEL_DIR, "net3_nor_last.pt")))
        att_probs = _forward(net1a, att_server, net3a, imgs)
        rows = _choose_images(victim, proxy, clean_probs, att_probs, labels, raw)
        for j, rec in enumerate(rows):
            img_path = os.path.join(CASE_DIR, f"{vname}_{j}.npy")
            np.save(img_path, raw[rec["index"]])
            rec["image_path"] = img_path
        rows_by_case[f"{vname}->{pname}"] = rows
        print(f"[M3] {vname}->{pname}: {len(rows)} images", flush=True)

    payload = {"model": TAG, "split": "SB=1", "dataset": "CIFAR-10",
               "cases": rows_by_case}
    json_path = os.path.join(CASE_DIR, "cases.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    _plot(rows_by_case, os.path.join(CASE_DIR, "case_study.png"),
          os.path.join(CASE_DIR, "case_study.pdf"))
    print(f"[M3] -> {json_path}\n[M3] -> {CASE_DIR}/case_study.pdf", flush=True)

    # Free the 10 GB cache; keep clusters (small), attacked servers, cases.
    shutil.rmtree(cache_dir, ignore_errors=True)
    free_gb = shutil.disk_usage("/root/autodl-tmp").free / 1e9
    print(f"[M3] freed cache; free={free_gb:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
