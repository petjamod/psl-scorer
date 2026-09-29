"""Compare the pretrained HCIILAB weights and the fine tuned model through the
app's exact preprocessing (aligned crop -> 256 -> center 224) on TRAIN and TEST.
The train/test gap is the leakage diagnostic. Writes runs/model_comparison.json."""
import json
import os

import numpy as np
import torch

from common import HERE, load_finetuned, load_pretrained, load_split, metrics
from finetune import aligned_256, evaluate


def main():
    xtr, xte = aligned_256("train"), aligned_256("test")
    ytr = np.array([r["score"] for r in load_split("train")])
    yte = np.array([r["score"] for r in load_split("test")])
    out = {}
    for name, model in (("pretrained", load_pretrained()),
                        ("finetuned", load_finetuned(os.path.join(HERE, "runs", "finetuned.pt")))):
        tr, _ = evaluate(model, xtr, ytr)
        te, _ = evaluate(model, xte, yte)
        out[name] = {"train_aligned": tr, "test_aligned": te, "gap_pearson": tr["pearson"] - te["pearson"]}
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(os.path.join(HERE, "runs", "model_comparison.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
