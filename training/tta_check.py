"""Does averaging with the horizontally flipped crop help? Test split, app preprocessing.
Writes runs/tta_check.json."""
import json
import os

import numpy as np
import torch

from common import HERE, load_finetuned, load_pretrained, load_split, metrics
from finetune import aligned_256, to_batch


@torch.no_grad()
def preds(model, x, flip):
    model.eval()
    out = []
    for i in range(0, len(x), 100):
        t = to_batch(x[i:i + 100], False)
        if flip:
            t = torch.flip(t, dims=[3])
        out.append(model(t).squeeze(1).numpy())
    return np.concatenate(out).astype(np.float64)


def main():
    xte = aligned_256("test")
    y = np.array([r["score"] for r in load_split("test")])
    res = {}
    for name, model in (("pretrained", load_pretrained()),
                        ("finetuned", load_finetuned(os.path.join(HERE, "runs", "finetuned.pt")))):
        a, b = preds(model, xte, False), preds(model, xte, True)
        res[name] = {"plain": metrics(a, y), "flip_avg": metrics((a + b) / 2, y),
                     "flip_mean_abs_disagreement": float(np.abs(a - b).mean())}
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(os.path.join(HERE, "runs", "tta_check.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
