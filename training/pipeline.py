"""Reference implementation of the exact preprocessing the web app runs.

Everything here is written so it can be ported line by line to JavaScript
(web/js/pipeline.js) and produce identical numbers:

  1. eye centers from MediaPipe landmarks (mean of the two eye corners)
  2. similarity transform: eyes level, inter eye distance and eye midpoint
     matched to the SCUT average (alignment.json)
  3. inverse bilinear warp to a 350x350 uint8 RGB crop (clamp to edge),
     after an antialiased pre-shrink when the face is much larger than SCUT's
  4. PIL-exact bilinear resize 350 -> 224 (fixed point, like Pillow's Resample.c)
  5. ImageNet normalization, RGB, CHW float32
"""
import math

import numpy as np

# MediaPipe face mesh indices of the eye corners. "Left" means image left
# (the subject's right eye).
L_OUTER, L_INNER = 33, 133
R_INNER, R_OUTER = 362, 263

FRAME = 350
INPUT = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PRECISION_BITS = 32 - 8 - 2


def eye_centers(pts_px):
    """pts_px: (478, 2) landmark pixel coordinates. Returns (left, right) as float64 arrays."""
    left = (pts_px[L_OUTER] + pts_px[L_INNER]) / 2.0
    right = (pts_px[R_INNER] + pts_px[R_OUTER]) / 2.0
    return np.asarray(left, np.float64), np.asarray(right, np.float64)


def similarity(left, right, tmpl):
    """Forward transform src -> dst as (a, b, tx, ty) with
    dst_x = a*x - b*y + tx, dst_y = b*x + a*y + ty."""
    dx, dy = right[0] - left[0], right[1] - left[1]
    dist = math.hypot(dx, dy)
    s = tmpl["eye_dist"] / dist
    ang = -math.atan2(dy, dx)
    a, b = s * math.cos(ang), s * math.sin(ang)
    mx, my = (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0
    tx = tmpl["eye_mid"][0] - (a * mx - b * my)
    ty = tmpl["eye_mid"][1] - (b * mx + a * my)
    return a, b, tx, ty, s


def pil_coeffs(in_size, out_size):
    """Pillow's precompute_coeffs + normalize_coeffs_8bpc for the bilinear filter."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 1.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    bounds = np.zeros((out_size, 2), np.int64)
    kk = np.zeros((out_size, ksize), np.int64)
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        ss = 1.0 / filterscale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size) - xmin
        w = np.zeros(xmax)
        for x in range(xmax):
            t = abs((x + xmin - center + 0.5) * ss)
            w[x] = 1.0 - t if t < 1.0 else 0.0
        ww = w.sum()
        if ww != 0.0:
            w = w / ww
        for x in range(xmax):
            kk[xx, x] = int(0.5 + w[x] * (1 << PRECISION_BITS)) if w[x] >= 0 else int(-0.5 + w[x] * (1 << PRECISION_BITS))
        bounds[xx] = (xmin, xmax)
    return bounds, kk


def _resample_axis0(img, out_size):
    """Resample along axis 0 of an (H, W, C) uint8 array, Pillow fixed point."""
    bounds, kk = pil_coeffs(img.shape[0], out_size)
    src = img.astype(np.int64)
    out = np.empty((out_size,) + img.shape[1:], np.uint8)
    for i in range(out_size):
        xmin, xmax = bounds[i]
        acc = np.full(img.shape[1:], 1 << (PRECISION_BITS - 1), np.int64)
        acc += np.tensordot(kk[i, :xmax], src[xmin:xmin + xmax], axes=(0, 0))
        out[i] = np.clip(acc >> PRECISION_BITS, 0, 255)
    return out


def resize_pil(img, out_w, out_h):
    """Bit exact port of PIL.Image.resize(..., Image.BILINEAR) for RGB uint8.
    Pillow runs the horizontal pass first, then the vertical one."""
    h, w = img.shape[:2]
    if out_w != w:
        img = _resample_axis0(img.transpose(1, 0, 2), out_w).transpose(1, 0, 2)
    if out_h != h:
        img = _resample_axis0(img, out_h)
    return np.ascontiguousarray(img)


def warp(img, a, b, tx, ty, size=FRAME):
    """Inverse bilinear warp with clamp to edge. img: (H, W, 3) uint8.
    Pixel centers sit at integer + 0.5 (the MediaPipe normalized-coordinate convention)."""
    h, w = img.shape[:2]
    det = a * a + b * b
    ia, ib = a / det, b / det  # inverse rotation+scale: [[ia, ib], [-ib, ia]]
    v, u = np.mgrid[0:size, 0:size].astype(np.float64)
    px, py = u + 0.5 - tx, v + 0.5 - ty
    sx = ia * px + ib * py - 0.5
    sy = -ib * px + ia * py - 0.5
    x0, y0 = np.floor(sx), np.floor(sy)
    fx, fy = sx - x0, sy - y0
    x0i, y0i = x0.astype(np.int64), y0.astype(np.int64)
    x1i, y1i = np.clip(x0i + 1, 0, w - 1), np.clip(y0i + 1, 0, h - 1)
    x0i, y0i = np.clip(x0i, 0, w - 1), np.clip(y0i, 0, h - 1)
    src = img.astype(np.float64)
    fx, fy = fx[..., None], fy[..., None]
    top = src[y0i, x0i] * (1 - fx) + src[y0i, x1i] * fx
    bot = src[y1i, x0i] * (1 - fx) + src[y1i, x1i] * fx
    val = top * (1 - fy) + bot * fy
    return np.clip(np.floor(val + 0.5), 0, 255).astype(np.uint8)


def inverse_point(a, b, tx, ty, x, y):
    det = a * a + b * b
    px, py = x - tx, y - ty
    return (a * px + b * py) / det, (-b * px + a * py) / det


def align(img, pts_px, tmpl, max_shrink=0.8):
    """Full alignment: returns the 350x350 uint8 crop and the transform (a, b, tx, ty, s).

    When the face has to be shrunk by more than max_shrink, the source region
    that maps into the frame is cut out and shrunk first with the antialiased
    Pillow filter, so the final bilinear warp runs at a scale close to 1 and
    does not alias. The web app does exactly the same (web/js/pipeline.js).
    """
    h, w = img.shape[:2]
    left, right = eye_centers(pts_px)
    a, b, tx, ty, s = similarity(left, right, tmpl)
    if s < max_shrink:
        corners = [inverse_point(a, b, tx, ty, x, y) for x, y in ((0, 0), (FRAME, 0), (0, FRAME), (FRAME, FRAME))]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        pad = 2.0 / s
        x0 = max(0, math.floor(min(xs) - pad))
        y0 = max(0, math.floor(min(ys) - pad))
        x1 = min(w, math.ceil(max(xs) + pad))
        y1 = min(h, math.ceil(max(ys) + pad))
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            cw, ch = x1 - x0, y1 - y0
            nw = max(1, math.floor(cw * s + 0.5))
            nh = max(1, math.floor(ch * s + 0.5))
            region = resize_pil(np.ascontiguousarray(img[y0:y1, x0:x1]), nw, nh)
            fx, fy = nw / cw, nh / ch
            pts2 = (pts_px - np.array([x0, y0], np.float64)) * np.array([fx, fy])
            left, right = eye_centers(pts2)
            a2, b2, tx2, ty2, _ = similarity(left, right, tmpl)
            return warp(region, a2, b2, tx2, ty2), (a, b, tx, ty, s)
    return warp(img, a, b, tx, ty), (a, b, tx, ty, s)


def to_tensor(crop350, resize="256crop"):
    """resize: "256crop" = resize to 256x256 then center crop 224 (the chosen variant),
    "224" = resize straight to 224x224."""
    if resize == "256crop":
        x = resize_pil(crop350, 256, 256)[16:240, 16:240]
    else:
        x = resize_pil(crop350, INPUT, INPUT)
    x = x.astype(np.float32)
    x = (x / np.float32(255.0) - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)
