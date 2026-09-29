"""Fine tune an ImageNet ResNet18 on the SCUT TRAIN split, CPU friendly.

Training images are the ALIGNED crops the web app produces (MediaPipe eyes ->
SCUT template -> 350x350), resized 350 -> 256 with the Pillow-exact filter.
Augmentation: random 224 crop, horizontal flip, mild brightness/contrast.
Evaluation: center 224 crop (identical to the app). conv1, layer1 and layer2
are frozen. The schedule is fixed in advance and the LAST epoch is kept, so
the test split is never used for model selection.

Usage: python finetune.py [minutes_budget]
Writes runs/finetuned.pt and runs/finetune_log.json.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from common import DATA, HERE, load_split, metrics
from pipeline import FRAME, MEAN, STD, align, resize_pil

torch.manual_seed(0)
np.random.seed(0)
RUNS = os.path.join(HERE, "runs")


def aligned_256(split):
    cache = os.path.join(RUNS, f"aligned256_{split}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    tmpl = json.load(open(os.path.join(HERE, "..", "web", "alignment.json")))
    lm = np.load(os.path.join(RUNS, f"landmarks_{split}.npz"), allow_pickle=True)
    pts = lm["pts"][:, :, :2].astype(np.float64) * FRAME
    rows = load_split(split)
    out = np.empty((len(rows), 256, 256, 3), np.uint8)
    for i, r in enumerate(rows):
        crop, _ = align(np.asarray(Image.open(r["path"]).convert("RGB")), pts[i], tmpl)
        out[i] = resize_pil(crop, 256, 256)
    np.save(cache, out)
    return out


MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
STD_T = torch.tensor(STD).view(1, 3, 1, 1)


def to_batch(u8, train):
    """u8: (B, 256, 256, 3) uint8 numpy -> normalized (B, 3, 224, 224) float tensor."""
    b = u8.shape[0]
    if train:
        out = np.empty((b, 224, 224, 3), np.uint8)
        for i in range(b):
            y, x = np.random.randint(0, 33, size=2)
            c = u8[i, y:y + 224, x:x + 224]
            out[i] = c[:, ::-1] if np.random.rand() < 0.5 else c
        t = torch.from_numpy(out).permute(0, 3, 1, 2).float() / 255.0
        bright = 1 + (torch.rand(b, 1, 1, 1) - 0.5) * 0.2
        contrast = 1 + (torch.rand(b, 1, 1, 1) - 0.5) * 0.2
        m = t.mean(dim=(1, 2, 3), keepdim=True)
        t = ((t - m) * contrast + m) * bright
        t = t.clamp(0, 1)
    else:
        t = torch.from_numpy(np.ascontiguousarray(u8[:, 16:240, 16:240])).permute(0, 3, 1, 2).float() / 255.0
    return (t - MEAN_T) / STD_T


def build_model(train_mean):
    model = torchvision.models.resnet18()
    model.load_state_dict(torch.load(os.path.join(DATA, "resnet18-imagenet.pth"), map_location="cpu"))
    model.fc = nn.Linear(512, 1)
    nn.init.normal_(model.fc.weight, std=0.01)
    nn.init.constant_(model.fc.bias, train_mean)
    frozen = [model.conv1, model.bn1, model.layer1, model.layer2]
    for mod in frozen:
        for p in mod.parameters():
            p.requires_grad = False
    return model, frozen


@torch.no_grad()
def evaluate(model, x, y, bs=100):
    model.eval()
    preds = [model(to_batch(x[i:i + bs], False)).squeeze(1).numpy() for i in range(0, len(x), bs)]
    return metrics(np.concatenate(preds), y), np.concatenate(preds)


def main(budget_min=55.0):
    t0 = time.time()
    xtr, xte = aligned_256("train"), aligned_256("test")
    ytr = np.array([r["score"] for r in load_split("train")], np.float32)
    yte = np.array([r["score"] for r in load_split("test")], np.float32)
    print(f"data ready in {time.time()-t0:.0f}s", xtr.shape, xte.shape, flush=True)

    model, frozen = build_model(float(ytr.mean()))
    params = [p for p in model.parameters() if p.requires_grad]
    bs = 32
    steps_per_epoch = len(xtr) // bs

    # Measure speed on a few steps, then fix the number of epochs for the budget.
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    t1 = time.time()
    model.train()
    for mod in frozen:
        mod.eval()
    for _ in range(5):
        idx = np.random.randint(0, len(xtr), bs)
        loss = nn.functional.mse_loss(model(to_batch(xtr[idx], True)).squeeze(1), torch.from_numpy(ytr[idx]))
        opt.zero_grad(); loss.backward(); opt.step()
    per_step = (time.time() - t1) / 5
    remaining = budget_min * 60 - (time.time() - t0)
    epochs = max(1, min(30, int(remaining / (per_step * steps_per_epoch * 1.08 + 25))))
    print(f"{per_step:.2f}s/step -> {epochs} epochs", flush=True)

    model, frozen = build_model(float(ytr.mean()))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    total = epochs * steps_per_epoch
    warm = steps_per_epoch // 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    log = {"epochs": epochs, "batch": bs, "per_step_s": per_step, "history": []}
    for ep in range(epochs):
        model.train()
        for mod in frozen:
            mod.eval()
        perm = np.random.permutation(len(xtr))
        tl = 0.0
        for k in range(steps_per_epoch):
            idx = np.sort(perm[k * bs:(k + 1) * bs])
            loss = nn.functional.mse_loss(model(to_batch(xtr[idx], True)).squeeze(1), torch.from_numpy(ytr[idx]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item()
        te, _ = evaluate(model, xte, yte)
        rec = {"epoch": ep + 1, "train_loss": tl / steps_per_epoch, "test": te, "elapsed_min": (time.time() - t0) / 60}
        log["history"].append(rec)
        print(json.dumps(rec), flush=True)
        torch.save(model.state_dict(), os.path.join(RUNS, "finetuned.pt"))
    tr, _ = evaluate(model, xtr, ytr)
    te, _ = evaluate(model, xte, yte)
    log["final"] = {"train": tr, "test": te}
    print("FINAL", json.dumps(log["final"]), flush=True)
    json.dump(log, open(os.path.join(RUNS, "finetune_log.json"), "w"), indent=2)


if __name__ == "__main__":
    main(*(float(a) for a in sys.argv[1:]))
