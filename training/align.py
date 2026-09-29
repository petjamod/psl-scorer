"""Phase 2: measure SCUT framing and check how much alignment costs.

Writes ../web/alignment.json (average eye geometry of the SCUT TRAIN images,
head pose reference and quality statistics) and runs/aligned_eval.json.
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image

from common import HERE, load_pretrained, load_split, metrics, predict
from pipeline import FRAME, align, eye_centers, to_tensor

WEB = os.path.join(HERE, "..", "web")


def euler_deg(m):
    """Yaw, pitch, roll in degrees from a 4x4 facial transformation matrix (row major)."""
    r = np.asarray(m, np.float64)[:3, :3]
    r = r / np.linalg.norm(r[:, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def laplacian_var(gray):
    g = gray.astype(np.float64)
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1])
    return float(lap.var())


def gray_of(rgb):
    r, g, b = rgb[..., 0].astype(np.float64), rgb[..., 1].astype(np.float64), rgb[..., 2].astype(np.float64)
    return 0.299 * r + 0.587 * g + 0.114 * b


def load_lm(split):
    d = np.load(os.path.join(HERE, "runs", f"landmarks_{split}.npz"), allow_pickle=True)
    return d


def pct(a, qs=(1, 5, 50, 95, 99)):
    return {str(q): float(np.percentile(a, q)) for q in qs}


def main():
    lm = load_lm("train")
    pts = lm["pts"][:, :, :2].astype(np.float64) * FRAME
    lefts, rights = zip(*(eye_centers(p) for p in pts))
    lefts, rights = np.array(lefts), np.array(rights)
    mids = (lefts + rights) / 2
    dists = np.hypot(*(rights - lefts).T)
    rolls = np.degrees(np.arctan2(*(rights - lefts).T[::-1]))
    poses = np.array([euler_deg(m) for m in lm["mats"]])
    bs_names = list(lm["bs_names"])
    bs = lm["bs"]
    tmpl = {
        "frame": FRAME,
        "eye_mid": [float(mids[:, 0].mean()), float(mids[:, 1].mean())],
        "eye_dist": float(dists.mean()),
        "left_eye_mean": [float(lefts[:, 0].mean()), float(lefts[:, 1].mean())],
        "right_eye_mean": [float(rights[:, 0].mean()), float(rights[:, 1].mean())],
        "eye_dist_std": float(dists.std()),
        "eye_mid_std": [float(mids[:, 0].std()), float(mids[:, 1].std())],
        "eye_roll_mean_deg": float(rolls.mean()),
        "landmarks": {"left_eye": [33, 133], "right_eye": [362, 263],
                      "note": "eye center = mean of the two corner landmarks; left = image left"},
        "pose_reference_deg": {"yaw": float(np.median(poses[:, 0])), "pitch": float(np.median(poses[:, 1])),
                               "roll": float(np.median(poses[:, 2]))},
        "n_images": int(len(pts)),
    }
    print(json.dumps(tmpl, indent=1))

    # SCUT distributions for the quality checks (computed on aligned train crops).
    train = load_split("train")
    lap, bright, dev = [], [], []
    for i, r in enumerate(train):
        img = np.asarray(Image.open(r["path"]).convert("RGB"))
        crop, _ = align(img, pts[i], tmpl)
        g = gray_of(crop)
        lap.append(laplacian_var(g))
        bright.append(float(g[100:300, 100:250].mean()))
    dev = np.abs(poses - np.array([tmpl["pose_reference_deg"][k] for k in ("yaw", "pitch", "roll")]))
    smile = np.maximum(bs[:, bs_names.index("mouthSmileLeft")], bs[:, bs_names.index("mouthSmileRight")])
    jaw = bs[:, bs_names.index("jawOpen")]
    tmpl["scut_quality"] = {
        "laplacian_var": pct(lap), "face_brightness": pct(bright),
        "abs_yaw_dev": pct(dev[:, 0]), "abs_pitch_dev": pct(dev[:, 1]), "abs_roll_dev": pct(dev[:, 2]),
        "smile": pct(smile), "jaw_open": pct(jaw),
        "frac_smile_gt_0.5": float((smile > 0.5).mean()), "frac_jaw_gt_0.5": float((jaw > 0.5).mean()),
    }
    print(json.dumps(tmpl["scut_quality"], indent=1))
    with open(os.path.join(WEB, "alignment.json"), "w") as f:
        json.dump(tmpl, f, indent=2)

    # How much does alignment cost on the TEST split?
    model = load_pretrained()
    lt = load_lm("test")
    tpts = lt["pts"][:, :, :2].astype(np.float64) * FRAME
    test = load_split("test")
    y = [r["score"] for r in test]
    arrs = []
    for i, r in enumerate(test):
        img = np.asarray(Image.open(r["path"]).convert("RGB"))
        crop, _ = align(img, tpts[i], tmpl)
        arrs.append(to_tensor(crop))
    m_aligned = metrics(predict(model, arrs), y)
    arrs = [to_tensor(np.asarray(Image.open(r["path"]).convert("RGB"))) for r in test]
    m_raw = metrics(predict(model, arrs), y)
    print("raw", m_raw, "\naligned", m_aligned)
    with open(os.path.join(HERE, "runs", "aligned_eval.json"), "w") as f:
        json.dump({"raw_numpy_pipeline": m_raw, "aligned": m_aligned}, f, indent=2)


if __name__ == "__main__":
    main()
