"""Evaluate the pretrained HCIILAB ResNet18 on the SCUT test split under every
plausible preprocessing variant. Writes runs/variants.json."""
import itertools
import json
import os
import time

from PIL import Image

from common import HERE, load_pretrained, load_split, metrics, predict, preprocess


def main():
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    model = load_pretrained()
    test = load_split("test")
    imgs = [Image.open(r["path"]).convert("RGB") for r in test]
    sizes = {im.size for im in imgs}
    print("image sizes:", sizes)
    y = [r["score"] for r in test]
    results = []
    for resize, norm, ch in itertools.product(["224", "256crop"], ["imagenet", "unit", "raw"], ["rgb", "bgr"]):
        t = time.time()
        arrs = [preprocess(im, resize, norm, ch) for im in imgs]
        m = metrics(predict(model, arrs), y)
        m.update(resize=resize, norm=norm, channels=ch)
        results.append(m)
        print(f"{resize:8s} {norm:9s} {ch}  PC={m['pearson']:.4f} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f}  ({time.time()-t:.0f}s)", flush=True)
    results.sort(key=lambda m: -m["pearson"])
    best = results[0]
    # Leakage check: the same model on the TRAIN split should score clearly higher
    # if (and only if) this split matches the one the weights were trained on.
    train = load_split("train")
    arrs = [preprocess(Image.open(r["path"]).convert("RGB"), best["resize"], best["norm"], best["channels"]) for r in train]
    train_m = metrics(predict(model, arrs), [r["score"] for r in train])
    print("best variant on TRAIN split:", train_m)
    with open(os.path.join(HERE, "runs", "variants.json"), "w") as f:
        json.dump({"test": results, "best_on_train": train_m}, f, indent=2)


if __name__ == "__main__":
    main()
