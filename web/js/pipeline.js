// Pure image pipeline: a line by line port of training/pipeline.py.
// Keep the two files in sync; the Playwright tests check that they agree.
//
// Images are plain objects { data: Uint8Array|Uint8ClampedArray, width, height, channels }
// with interleaved channels (3 = RGB, 4 = RGBA; alpha is ignored).

export const FRAME = 350;
export const INPUT = 224;
const PRECISION_BITS = 32 - 8 - 2;
const MEAN = [0.485, 0.456, 0.406].map(Math.fround);
const STD = [0.229, 0.224, 0.225].map(Math.fround);

// MediaPipe face mesh eye corners. "Left" means image left (the subject's right eye).
const L_OUTER = 33, L_INNER = 133, R_INNER = 362, R_OUTER = 263;

export function eyeCenters(pts) {
  // pts: array of [x, y] pixel coordinates (478 entries)
  const left = [(pts[L_OUTER][0] + pts[L_INNER][0]) / 2, (pts[L_OUTER][1] + pts[L_INNER][1]) / 2];
  const right = [(pts[R_INNER][0] + pts[R_OUTER][0]) / 2, (pts[R_INNER][1] + pts[R_OUTER][1]) / 2];
  return [left, right];
}

// Forward transform src -> dst: x' = a*x - b*y + tx, y' = b*x + a*y + ty
export function similarity(left, right, tmpl) {
  const dx = right[0] - left[0], dy = right[1] - left[1];
  const dist = Math.hypot(dx, dy);
  const s = tmpl.eye_dist / dist;
  const ang = -Math.atan2(dy, dx);
  const a = s * Math.cos(ang), b = s * Math.sin(ang);
  const mx = (left[0] + right[0]) / 2, my = (left[1] + right[1]) / 2;
  const tx = tmpl.eye_mid[0] - (a * mx - b * my);
  const ty = tmpl.eye_mid[1] - (b * mx + a * my);
  return { a, b, tx, ty, s, eyeDist: dist };
}

function inversePoint(T, x, y) {
  const det = T.a * T.a + T.b * T.b;
  const px = x - T.tx, py = y - T.ty;
  return [(T.a * px + T.b * py) / det, (-T.b * px + T.a * py) / det];
}

// Pillow's precompute_coeffs + normalize_coeffs_8bpc (bilinear filter).
const coeffCache = new Map();
function pilCoeffs(inSize, outSize) {
  const key = inSize + 'x' + outSize;
  if (coeffCache.has(key)) return coeffCache.get(key);
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1.0);
  const support = 1.0 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Float64Array(outSize * ksize); // integer valued
  const w = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1.0 / filterscale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const t = Math.abs((x + xmin - center + 0.5) * ss);
      w[x] = t < 1.0 ? 1.0 - t : 0.0;
      ww += w[x];
    }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0.0 ? w[x] / ww : w[x];
      kk[xx * ksize + x] = v >= 0 ? Math.trunc(0.5 + v * (1 << PRECISION_BITS)) : Math.trunc(-0.5 + v * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  const res = { bounds, kk, ksize };
  coeffCache.set(key, res);
  return res;
}

const HALF = 1 << (PRECISION_BITS - 1);
const DIV = 2 ** PRECISION_BITS;
function clip8(acc) {
  const v = Math.floor(acc / DIV);
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

// Bit exact port of PIL.Image.resize(..., Image.BILINEAR): horizontal pass, then vertical.
// Output is always 3 channel RGB.
export function resizePIL(img, outW, outH) {
  const { width: w, height: h, channels: ch, data } = img;
  let cur = { data, width: w, height: h, channels: ch };
  if (outW !== w) {
    const { bounds, kk, ksize } = pilCoeffs(w, outW);
    const out = new Uint8Array(outW * h * 3);
    for (let y = 0; y < h; y++) {
      const row = y * w * ch;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1], k0 = xx * ksize;
        let s0 = HALF, s1 = HALF, s2 = HALF;
        for (let x = 0; x < xmax; x++) {
          const k = kk[k0 + x], p = row + (x + xmin) * ch;
          s0 += data[p] * k; s1 += data[p + 1] * k; s2 += data[p + 2] * k;
        }
        const o = (y * outW + xx) * 3;
        out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2);
      }
    }
    cur = { data: out, width: outW, height: h, channels: 3 };
  }
  if (outH !== h) {
    const { bounds, kk, ksize } = pilCoeffs(h, outH);
    const src = cur.data, cw = cur.width, cc = cur.channels;
    const out = new Uint8Array(cw * outH * 3);
    for (let yy = 0; yy < outH; yy++) {
      const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1], k0 = yy * ksize;
      for (let x = 0; x < cw; x++) {
        let s0 = HALF, s1 = HALF, s2 = HALF;
        for (let y = 0; y < ymax; y++) {
          const k = kk[k0 + y], p = ((y + ymin) * cw + x) * cc;
          s0 += src[p] * k; s1 += src[p + 1] * k; s2 += src[p + 2] * k;
        }
        const o = (yy * cw + x) * 3;
        out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2);
      }
    }
    cur = { data: out, width: cw, height: outH, channels: 3 };
  }
  if (cur.channels !== 3) cur = toRGB(cur);
  return cur;
}

export function toRGB(img) {
  if (img.channels === 3) return img;
  const n = img.width * img.height, out = new Uint8Array(n * 3), d = img.data, c = img.channels;
  for (let i = 0; i < n; i++) { out[i * 3] = d[i * c]; out[i * 3 + 1] = d[i * c + 1]; out[i * 3 + 2] = d[i * c + 2]; }
  return { data: out, width: img.width, height: img.height, channels: 3 };
}

function cropImage(img, x0, y0, cw, ch) {
  const out = new Uint8Array(cw * ch * 3), c = img.channels, d = img.data;
  for (let y = 0; y < ch; y++) {
    let p = ((y + y0) * img.width + x0) * c, o = y * cw * 3;
    for (let x = 0; x < cw; x++, p += c, o += 3) { out[o] = d[p]; out[o + 1] = d[p + 1]; out[o + 2] = d[p + 2]; }
  }
  return { data: out, width: cw, height: ch, channels: 3 };
}

// Inverse bilinear warp with clamp to edge; pixel centers at integer + 0.5.
export function warp(img, T, size = FRAME) {
  const { width: w, height: h, channels: c, data } = img;
  const det = T.a * T.a + T.b * T.b;
  const ia = T.a / det, ib = T.b / det;
  const out = new Uint8Array(size * size * 3);
  for (let v = 0; v < size; v++) {
    for (let u = 0; u < size; u++) {
      const px = u + 0.5 - T.tx, py = v + 0.5 - T.ty;
      const sx = ia * px + ib * py - 0.5;
      const sy = -ib * px + ia * py - 0.5;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const fx = sx - x0, fy = sy - y0;
      const xa = x0 < 0 ? 0 : x0 > w - 1 ? w - 1 : x0;
      const ya = y0 < 0 ? 0 : y0 > h - 1 ? h - 1 : y0;
      const xb = x0 + 1 < 0 ? 0 : x0 + 1 > w - 1 ? w - 1 : x0 + 1;
      const yb = y0 + 1 < 0 ? 0 : y0 + 1 > h - 1 ? h - 1 : y0 + 1;
      const p00 = (ya * w + xa) * c, p01 = (ya * w + xb) * c, p10 = (yb * w + xa) * c, p11 = (yb * w + xb) * c;
      const o = (v * size + u) * 3;
      for (let k = 0; k < 3; k++) {
        const top = data[p00 + k] * (1 - fx) + data[p01 + k] * fx;
        const bot = data[p10 + k] * (1 - fx) + data[p11 + k] * fx;
        let val = Math.floor(top * (1 - fy) + bot * fy + 0.5);
        out[o + k] = val < 0 ? 0 : val > 255 ? 255 : val;
      }
    }
  }
  return { data: out, width: size, height: size, channels: 3 };
}

// Full alignment. pts are landmark pixel coordinates in `img`.
// Returns { crop, T } where T is the transform from the original image.
export function align(img, pts, tmpl, maxShrink = 0.8) {
  const [left, right] = eyeCenters(pts);
  const T = similarity(left, right, tmpl);
  if (T.s < maxShrink) {
    const corners = [[0, 0], [FRAME, 0], [0, FRAME], [FRAME, FRAME]].map(([x, y]) => inversePoint(T, x, y));
    const xs = corners.map((p) => p[0]), ys = corners.map((p) => p[1]);
    const pad = 2.0 / T.s;
    const x0 = Math.max(0, Math.floor(Math.min(...xs) - pad));
    const y0 = Math.max(0, Math.floor(Math.min(...ys) - pad));
    const x1 = Math.min(img.width, Math.ceil(Math.max(...xs) + pad));
    const y1 = Math.min(img.height, Math.ceil(Math.max(...ys) + pad));
    if (x1 - x0 >= 2 && y1 - y0 >= 2) {
      const cw = x1 - x0, ch = y1 - y0;
      const nw = Math.max(1, Math.floor(cw * T.s + 0.5));
      const nh = Math.max(1, Math.floor(ch * T.s + 0.5));
      const region = resizePIL(cropImage(img, x0, y0, cw, ch), nw, nh);
      const fx = nw / cw, fy = nh / ch;
      const pts2 = [];
      for (const i of [L_OUTER, L_INNER, R_INNER, R_OUTER]) pts2[i] = [(pts[i][0] - x0) * fx, (pts[i][1] - y0) * fy];
      const [l2, r2] = eyeCenters(pts2);
      const T2 = similarity(l2, r2, tmpl);
      return { crop: warp(region, T2), T };
    }
  }
  return { crop: warp(img, T), T };
}

// 350x350 RGB crop -> normalized CHW Float32Array for the network.
// variant "256crop": resize to 256 then center crop 224; "224": resize straight to 224.
export function toTensor(crop, variant = '256crop') {
  let src, off, side;
  if (variant === '256crop') { src = resizePIL(crop, 256, 256); off = 16; side = 256; }
  else { src = resizePIL(crop, INPUT, INPUT); off = 0; side = INPUT; }
  const out = new Float32Array(3 * INPUT * INPUT);
  const plane = INPUT * INPUT;
  const d = src.data;
  for (let y = 0; y < INPUT; y++) {
    for (let x = 0; x < INPUT; x++) {
      const p = ((y + off) * side + (x + off)) * 3, o = y * INPUT + x;
      for (let k = 0; k < 3; k++) {
        const v = Math.fround(d[p + k] / 255);
        out[k * plane + o] = Math.fround(Math.fround(v - MEAN[k]) / STD[k]);
      }
    }
  }
  return out;
}

// ---- quality measures (same definitions as training/align.py) ----

export function grayOf(img) {
  const n = img.width * img.height, g = new Float64Array(n), d = img.data, c = img.channels;
  for (let i = 0; i < n; i++) g[i] = 0.299 * d[i * c] + 0.587 * d[i * c + 1] + 0.114 * d[i * c + 2];
  return g;
}

export function laplacianVar(gray, w, h) {
  let sum = 0, sum2 = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const l = gray[i - w] + gray[i + w] + gray[i - 1] + gray[i + 1] - 4 * gray[i];
      sum += l; sum2 += l * l; n++;
    }
  }
  const m = sum / n;
  return sum2 / n - m * m;
}

// Mean luma of the central face region of the aligned crop.
export function faceBrightness(gray, w) {
  let s = 0, n = 0;
  for (let y = 100; y < 300; y++) for (let x = 100; x < 250; x++) { s += gray[y * w + x]; n++; }
  return s / n;
}

// Yaw, pitch, roll (degrees) from MediaPipe's 4x4 facial transformation matrix.
export function eulerDeg(data) {
  // MediaPipe JS returns the matrix column major; detect it from where the (always
  // clearly non zero) z translation sits, so either layout works.
  const colMajor = Math.abs(data[14]) > Math.abs(data[11]);
  const m = (i, j) => (colMajor ? data[j * 4 + i] : data[i * 4 + j]);
  const sc = Math.hypot(m(0, 0), m(1, 0), m(2, 0));
  const r = (i, j) => m(i, j) / sc;
  const deg = 180 / Math.PI;
  const pitch = Math.atan2(r(2, 1), r(2, 2)) * deg;
  const yaw = Math.asin(Math.max(-1, Math.min(1, -r(2, 0)))) * deg;
  const roll = Math.atan2(r(1, 0), r(0, 0)) * deg;
  return { yaw, pitch, roll };
}
