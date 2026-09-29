"""Extract SCUT-FBP5500 images and labels from the Hugging Face parquet mirror.

Downloads MnLgt/scut-fbp5500 (3300 train / 2200 test, the official 60/40 split)
if the parquet files are missing, writes every image to data/images/<name>
and the labels to data/{train,test}.csv.
"""
import csv
import os
import subprocess

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
IMG = os.path.join(DATA, "images")
BASE = "https://huggingface.co/datasets/MnLgt/scut-fbp5500/resolve/main/data/"
WEIGHTS = "https://huggingface.co/Gustrd/SCUT-FBP5500-PyTorch-Model/resolve/main/resnet18_py3.pth"
LANDMARKER = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")


def fetch(url, dst):
    if not os.path.exists(dst):
        print("downloading", url)
        subprocess.check_call(["curl", "-sSfL", "-o", dst, url])


def main():
    os.makedirs(IMG, exist_ok=True)
    fetch(WEIGHTS, os.path.join(DATA, "resnet18_py3.pth"))
    fetch(LANDMARKER, os.path.join(DATA, "face_landmarker.task"))
    for split in ("train", "test"):
        pq_path = os.path.join(DATA, f"{split}.parquet")
        fetch(BASE + f"{split}-00000-of-00001.parquet", pq_path)
        table = pq.read_table(pq_path)
        rows = []
        for batch in table.to_batches(max_chunksize=256):
            for r in batch.to_pylist():
                name = r["image_name"]
                out = os.path.join(IMG, os.path.splitext(name)[0] + ".png")
                if not os.path.exists(out):
                    with open(out, "wb") as f:
                        f.write(r["image"]["bytes"])
                rows.append((name, f"{r['beauty_score']:.6f}", r["gender"], r["race"]))
        with open(os.path.join(DATA, f"{split}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "score", "gender", "race"])
            w.writerows(rows)
        print(split, len(rows))


if __name__ == "__main__":
    main()
