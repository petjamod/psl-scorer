"""Shared helpers: dataset access, model loading and preprocessing variants."""
import csv
import os

import numpy as np
import torch
import torchvision
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
IMG = os.path.join(DATA, "images")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

torch.set_num_threads(os.cpu_count() or 4)


def load_split(split):
    with open(os.path.join(DATA, f"{split}.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["score"] = float(r["score"])
        r["path"] = os.path.join(IMG, os.path.splitext(r["name"])[0] + ".png")
    return rows


def remap_hciilab(state_dict):
    """Map the HCIILAB Nets.py ResNet key names onto torchvision's resnet18."""
    out = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if k.startswith("group1."):
            k = k[len("group1."):]
        k = k.replace(".group1.", ".")
        k = k.replace("group2.fullyconnected.", "fc.")
        out[k] = v
    return out


def load_pretrained(path=os.path.join(DATA, "resnet18_py3.pth")):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    model = torchvision.models.resnet18(num_classes=1)
    model.load_state_dict(remap_hciilab(sd), strict=True)
    return model.eval()


def load_finetuned(path):
    model = torchvision.models.resnet18(num_classes=1)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def preprocess(img, resize="224", norm="imagenet", channels="rgb"):
    """img: PIL RGB image (any size). Returns float32 CHW numpy array.

    resize: "224" = resize to 224x224, "256crop" = resize to 256x256 then center crop 224.
    norm: "imagenet" = (x/255 - mean)/std, "unit" = x/255, "raw" = x (0..255).
    channels: "rgb" or "bgr" (order fed to the network).
    """
    if resize == "224":
        img = img.resize((224, 224), Image.BILINEAR)
    elif resize == "256crop":
        img = img.resize((256, 256), Image.BILINEAR).crop((16, 16, 240, 240))
    else:
        raise ValueError(resize)
    x = np.asarray(img, dtype=np.float32)
    if channels == "bgr":
        x = x[:, :, ::-1]
    if norm == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        if channels == "bgr":
            mean, std = mean[::-1], std[::-1]
        x = (x / 255.0 - mean) / std
    elif norm == "unit":
        x = x / 255.0
    elif norm != "raw":
        raise ValueError(norm)
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)


@torch.no_grad()
def predict(model, arrays, batch=64):
    out = []
    for i in range(0, len(arrays), batch):
        x = torch.from_numpy(np.stack(arrays[i:i + batch]))
        out.append(model(x).squeeze(1).numpy())
    return np.concatenate(out).astype(np.float64)


def metrics(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    pc = float(np.corrcoef(pred, true)[0, 1])
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return {"pearson": pc, "mae": mae, "rmse": rmse, "n": int(len(pred))}
