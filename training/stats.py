"""Score statistics of the TRAIN split labels per gender -> ../web/stats.json."""
import json
import os

import numpy as np

from common import HERE, load_split


def summarize(scores):
    s = np.asarray(scores, np.float64)
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "max": float(s.max()),
        "percentiles": {str(p): float(np.percentile(s, p)) for p in range(1, 100)},
    }


def main():
    train = load_split("train")
    out = {
        "source": "SCUT-FBP5500 official 60/40 split, TRAIN labels (mean of 60 raters, 1 to 5)",
        "male": summarize([r["score"] for r in train if r["gender"] == "Male"]),
        "female": summarize([r["score"] for r in train if r["gender"] == "Female"]),
        "all": summarize([r["score"] for r in train]),
    }
    for k in ("male", "female", "all"):
        print(k, out[k]["n"], round(out[k]["mean"], 4), round(out[k]["std"], 4))
    json.dump(out, open(os.path.join(HERE, "..", "web", "stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
