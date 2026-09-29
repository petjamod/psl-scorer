"""Build the Phase 4 verification fixtures in ../tests/out/fixtures and the
Python reference predictions for them (using the ONNX model the app ships).
"""
import io
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

from align import euler_deg
from common import HERE, load_split
from pipeline import FRAME, align, eye_centers, to_tensor

ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "tests", "out", "fixtures")
WEB = os.path.join(ROOT, "web")


def main():
    for d in ("jpeg85", "jpeg70", "quality", "tensors"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    cfg = json.load(open(os.path.join(WEB, "config.json")))
    tmpl = json.load(open(os.path.join(WEB, "alignment.json")))
    variant = cfg["model"]["preprocess"]
    sess = ort.InferenceSession(os.path.join(WEB, cfg["model"]["file"]), providers=["CPUExecutionProvider"])
    run = lambda x: float(sess.run(None, {"input": x[None]})[0][0, 0])

    test = load_split("test")
    lm = np.load(os.path.join(HERE, "runs", "landmarks_test.npz"), allow_pickle=True)
    pts = lm["pts"][:, :, :2].astype(np.float64) * FRAME
    idx = np.sort(np.random.RandomState(0).choice(len(test), 300, replace=False))
    e2e = []
    for i in idx:
        r = test[i]
        img = np.asarray(Image.open(r["path"]).convert("RGB"))
        crop, _ = align(img, pts[i], tmpl)
        l, rr = eye_centers(pts[i])
        e2e.append({"name": r["name"], "png": os.path.splitext(r["name"])[0] + ".png", "score": r["score"],
                    "gender": r["gender"], "py_raw_aligned": run(to_tensor(crop, variant)),
                    "py_raw_unaligned": run(to_tensor(img, variant)), "py_eyes": [l.tolist(), rr.tolist()]})
    json.dump({"variant": variant, "model": cfg["model"]["id"], "items": e2e}, open(os.path.join(OUT, "e2e.json"), "w"), indent=1)

    # Tensor parity fixtures: Python tensors of the raw (unaligned) images for the first 20.
    for it in e2e[:20]:
        img = np.asarray(Image.open(os.path.join(HERE, "data", "images", it["png"])).convert("RGB"))
        to_tensor(img, variant).astype("<f4").tofile(os.path.join(OUT, "tensors", it["png"] + ".f32"))

    # JPEG robustness: first 50 re-encoded at quality 85 and 70.
    for it in e2e[:50]:
        im = Image.open(os.path.join(HERE, "data", "images", it["png"])).convert("RGB")
        base = os.path.splitext(it["png"])[0]
        im.save(os.path.join(OUT, "jpeg85", base + ".jpg"), quality=85)
        im.save(os.path.join(OUT, "jpeg70", base + ".jpg"), quality=70)

    # Quality gate fixtures.
    q = os.path.join(OUT, "quality")
    names = [r["name"] for r in test]
    bs_names = list(lm["bs_names"])
    smile = np.maximum(lm["bs"][:, bs_names.index("mouthSmileLeft")], lm["bs"][:, bs_names.index("mouthSmileRight")])
    ref = tmpl["pose_reference_deg"]
    yaw_dev = np.array([abs(euler_deg(m)[0] - ref["yaw"]) for m in lm["mats"]])
    neutral = [i for i in np.argsort(smile) if yaw_dev[i] < 5][:3]
    src = lambda i: Image.open(test[i]["path"]).convert("RGB")
    manifest = {}

    a = src(neutral[0])
    a.save(os.path.join(q, "neutral.png"))
    manifest["neutral"] = {"source": names[neutral[0]], "expect": []}

    bg = tuple(int(v) for v in np.asarray(a)[:10].reshape(-1, 3).mean(0))
    a.rotate(30, resample=Image.BICUBIC, expand=True, fillcolor=bg).save(os.path.join(q, "rotated30.png"))
    manifest["rotated30"] = {"source": names[neutral[0]], "expect": ["pose"]}

    i = int(np.argmax(smile))
    src(i).save(os.path.join(q, "smiling.png"))
    manifest["smiling"] = {"source": names[i], "smile": float(smile[i]), "expect": ["expression"]}

    i = int(np.argmax(yaw_dev))
    src(i).save(os.path.join(q, "turned.png"))
    manifest["turned"] = {"source": names[i], "python_yaw_dev": float(yaw_dev[i]), "expect": ["pose"] if yaw_dev[i] > 15 else []}

    two = Image.new("RGB", (700, 350))
    two.paste(src(neutral[1]), (0, 0)); two.paste(src(neutral[2]), (350, 0))
    two.save(os.path.join(q, "two_faces.png"))
    manifest["two_faces"] = {"source": [names[neutral[1]], names[neutral[2]]], "expect_status": "multiface"}

    a.resize((120, 120), Image.BILINEAR).save(os.path.join(q, "small.png"))
    manifest["small"] = {"source": names[neutral[0]], "expect": ["small_face"]}

    Image.fromarray((np.asarray(a).astype(np.float32) * 0.25).astype(np.uint8)).save(os.path.join(q, "dark.png"))
    manifest["dark"] = {"source": names[neutral[0]], "expect": ["lighting"]}

    Image.new("RGB", (400, 300), (120, 130, 140)).save(os.path.join(q, "noface.png"))
    manifest["noface"] = {"expect_status": "noface"}

    # EXIF orientation: pixels stored rotated 90 degrees CCW, tag 6 says "rotate 90 CW to display".
    up = a.copy()
    up.save(os.path.join(q, "exif_upright.jpg"), quality=95)
    ex = Image.Exif(); ex[0x0112] = 6
    up.rotate(90, expand=True).save(os.path.join(q, "exif_rot6.jpg"), quality=95, exif=ex.tobytes())
    manifest["exif_rot6"] = {"source": names[neutral[0]], "compare_to": "exif_upright.jpg", "expect": []}

    # Large photo: face upscaled 4x inside a 3000x2200 frame, exercises the antialiased
    # pre-shrink path. Python reference computed here with Python MediaPipe.
    from landmarks import make_landmarker
    import mediapipe as mp
    big = Image.new("RGB", (3000, 2200), (128, 128, 128))
    big.paste(a.resize((1400, 1400), Image.BICUBIC), (900, 500))
    big.save(os.path.join(q, "big.png"))
    res = make_landmarker().detect(mp.Image.create_from_file(os.path.join(q, "big.png")))
    bp = np.array([(p.x * 3000, p.y * 2200) for p in res.face_landmarks[0]], np.float64)
    bcrop, bT = align(np.asarray(big), bp, tmpl)
    Image.fromarray(bcrop).save(os.path.join(OUT, "big_crop_python.png"))
    manifest["big"] = {"source": names[neutral[0]], "expect": [], "py_raw": run(to_tensor(bcrop, variant)),
                       "py_scale": bT[4], "raw_of_original_350": run(to_tensor(align(np.asarray(a), pts[neutral[0]], tmpl)[0], variant))}

    json.dump(manifest, open(os.path.join(q, "manifest.json"), "w"), indent=1)
    print("fixtures:", len(e2e), "e2e;", list(manifest))


if __name__ == "__main__":
    main()
