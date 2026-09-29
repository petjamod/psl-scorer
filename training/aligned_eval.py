"""Evaluate the pretrained model on aligned SCUT test crops for both resize variants."""
import json
import os
import sys

import numpy as np
from PIL import Image

from common import HERE, load_pretrained, load_split, metrics, predict
from pipeline import FRAME, align, to_tensor

WEB = os.path.join(HERE, "..", "web")


def main(variants=("256crop", "224")):
    tmpl = json.load(open(os.path.join(WEB, "alignment.json")))
    model = load_pretrained()
    lt = np.load(os.path.join(HERE, "runs", "landmarks_test.npz"), allow_pickle=True)
    tpts = lt["pts"][:, :, :2].astype(np.float64) * FRAME
    test = load_split("test")
    y = [r["score"] for r in test]
    crops = [align(np.asarray(Image.open(r["path"]).convert("RGB")), tpts[i], tmpl)[0] for i, r in enumerate(test)]
    raws = [np.asarray(Image.open(r["path"]).convert("RGB")) for r in test]
    out = {}
    for v in variants:
        out[v] = {"aligned": metrics(predict(model, [to_tensor(c, v) for c in crops]), y),
                  "raw": metrics(predict(model, [to_tensor(c, v) for c in raws]), y)}
        print(v, out[v], flush=True)
    json.dump(out, open(os.path.join(HERE, "runs", "aligned_eval.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
