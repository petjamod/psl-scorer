"""Export the chosen model to ONNX (fp32) and verify onnxruntime against PyTorch.

Usage: python export_onnx.py [pretrained|path/to/finetuned.pt] [flip]
With "flip" the exported graph averages the prediction of the crop and of its
mirror image (same weights, two passes).
Writes ../web/models/scut_resnet18.onnx and runs/onnx_check.json.
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from common import HERE, load_finetuned, load_pretrained, load_split, predict
from pipeline import to_tensor

OUT = os.path.join(HERE, "..", "web", "models", "scut_resnet18.onnx")


class FlipAverage(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return (self.m(x) + self.m(torch.flip(x, dims=[3]))) * 0.5


def main(which="pretrained", flip=""):
    model = load_pretrained() if which == "pretrained" else load_finetuned(which)
    if flip == "flip":
        model = FlipAverage(model).eval()
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(model, dummy, OUT, input_names=["input"], output_names=["score"],
                      dynamic_axes={"input": {0: "batch"}, "score": {0: "batch"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    size = os.path.getsize(OUT)
    print("wrote", OUT, size, "bytes")

    test = load_split("test")[:200]
    arrs = [to_tensor(np.asarray(Image.open(r["path"]).convert("RGB"))) for r in test]
    pt = predict(model, arrs)
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    ox = np.array([sess.run(None, {"input": a[None]})[0][0, 0] for a in arrs], dtype=np.float64)
    diff = np.abs(pt - ox)
    res = {"n": len(arrs), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
           "onnx_bytes": size, "source": which, "flip_average": flip == "flip", "pass": bool(diff.max() < 0.005)}
    print(res)
    json.dump(res, open(os.path.join(HERE, "runs", "onnx_check.json"), "w"), indent=2)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
