"""Evaluate the shipped ONNX model on the full SCUT TEST split through the exact
app preprocessing (aligned crop -> 256 -> center 224), with breakdowns.
Writes runs/final_eval.json."""
import json
import os

import numpy as np
import onnxruntime as ort

from common import HERE, load_split, metrics
from finetune import aligned_256
from pipeline import MEAN, STD

WEB = os.path.join(HERE, "..", "web")


def main():
    cfg = json.load(open(os.path.join(WEB, "config.json")))
    assert cfg["model"]["preprocess"] == "256crop"
    sess = ort.InferenceSession(os.path.join(WEB, cfg["model"]["file"]), providers=["CPUExecutionProvider"])
    rows = load_split("test")
    x = aligned_256("test")
    preds = []
    for i in range(0, len(x), 50):
        b = x[i:i + 50, 16:240, 16:240].astype(np.float32)
        b = ((b / np.float32(255.0) - MEAN) / STD).transpose(0, 3, 1, 2)
        preds.append(sess.run(None, {"input": np.ascontiguousarray(b)})[0][:, 0])
    p = np.concatenate(preds).astype(np.float64)
    y = np.array([r["score"] for r in rows])
    lm = np.load(os.path.join(HERE, "runs", "landmarks_test.npz"), allow_pickle=True)
    names = list(lm["bs_names"])
    smile = np.maximum(lm["bs"][:, names.index("mouthSmileLeft")], lm["bs"][:, names.index("mouthSmileRight")])
    out = {"model": cfg["model"]["id"], "all": metrics(p, y)}
    for key, vals in (("gender", ("Male", "Female")), ("race", ("Asian", "Caucasian"))):
        for v in vals:
            m = np.array([r[key] == v for r in rows])
            out[f"{key}={v}"] = metrics(p[m], y[m])
    for lab, m in (("smile>0.5", smile > 0.5), ("smile<=0.5", smile <= 0.5)):
        out[lab] = metrics(p[m], y[m])
        out[lab]["mean_residual_pred_minus_label"] = float((p[m] - y[m]).mean())
    out["prediction_std"] = float(p.std()); out["label_std"] = float(y.std())
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "runs", "final_eval.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
