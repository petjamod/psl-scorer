"""Run MediaPipe Face Landmarker over SCUT images and cache the results.

Writes runs/landmarks_<split>.npz with, per image: number of faces found,
478 normalized landmarks (x, y, z), 52 blendshape scores and the 4x4 facial
transformation matrix of the first face.
"""
import os
import sys

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from common import DATA, HERE, load_split


def make_landmarker(num_faces=2):
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(DATA, "face_landmarker.task")),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=num_faces,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def run(split):
    rows = load_split(split)
    lm = make_landmarker()
    n = len(rows)
    faces = np.zeros(n, np.int32)
    pts = np.full((n, 478, 3), np.nan, np.float32)
    bs = np.full((n, 52), np.nan, np.float32)
    mats = np.full((n, 4, 4), np.nan, np.float32)
    for i, r in enumerate(rows):
        res = lm.detect(mp.Image.create_from_file(r["path"]))
        faces[i] = len(res.face_landmarks)
        if faces[i]:
            pts[i] = [(p.x, p.y, p.z) for p in res.face_landmarks[0]]
            bs[i] = [c.score for c in res.face_blendshapes[0]]
            mats[i] = np.asarray(res.facial_transformation_matrixes[0])
        if i % 500 == 0:
            print(split, i, flush=True)
    names = np.array([r["name"] for r in rows])
    bs_names = np.array([c.category_name for c in res.face_blendshapes[0]]) if faces[-1] else None
    np.savez_compressed(os.path.join(HERE, "runs", f"landmarks_{split}.npz"),
                        names=names, faces=faces, pts=pts, bs=bs, mats=mats, bs_names=bs_names)
    print(split, "faces histogram:", np.bincount(faces).tolist())


if __name__ == "__main__":
    for s in sys.argv[1:] or ["train", "test"]:
        run(s)
