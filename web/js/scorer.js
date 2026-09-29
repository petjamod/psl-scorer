// Loads the models (self hosted, pinned) and runs the full per photo pipeline:
// decode (EXIF aware) -> Face Landmarker -> quality checks -> align -> ONNX score.
import * as ort from '../vendor/ort/ort.wasm.min.mjs';
import { FilesetResolver, FaceLandmarker } from '../vendor/mediapipe/vision_bundle.mjs';
import { align, toTensor, grayOf, laplacianVar, faceBrightness, eulerDeg, eyeCenters } from './pipeline.js';

const base = new URL('../', import.meta.url);
const url = (p) => new URL(p, base).href;

async function loadJSON(p) {
  const r = await fetch(url(p));
  if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
  return r.json();
}

async function fetchWithProgress(p, expected, onProgress) {
  const r = await fetch(url(p));
  if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
  const total = Number(r.headers.get('content-length')) || expected || 0;
  if (!r.body || !r.body.getReader) {
    const buf = new Uint8Array(await r.arrayBuffer());
    onProgress?.(buf.length, buf.length);
    return buf;
  }
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(got, Math.max(total, got));
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

export async function sha256Hex(buf) {
  const d = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

// Decode with EXIF orientation applied. Returns { image: {data,width,height,channels}, imageData }
export async function decode(blob, maxPixels = 16e6) {
  let src;
  try {
    src = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  } catch (e) {
    // Fallback (e.g. formats createImageBitmap rejects): let an <img> decode it.
    src = await new Promise((resolve, reject) => {
      const img = new Image();
      const u = URL.createObjectURL(blob);
      img.onload = () => { URL.revokeObjectURL(u); resolve(img); };
      img.onerror = () => { URL.revokeObjectURL(u); reject(new Error('Could not decode this image')); };
      img.src = u;
    });
  }
  let w = src.width || src.naturalWidth, h = src.height || src.naturalHeight;
  let f = 1;
  if (w * h > maxPixels) f = Math.sqrt(maxPixels / (w * h));
  const cw = Math.max(1, Math.round(w * f)), ch = Math.max(1, Math.round(h * f));
  const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(cw, ch) : Object.assign(document.createElement('canvas'), { width: cw, height: ch });
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (f !== 1) { ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high'; }
  ctx.drawImage(src, 0, 0, cw, ch);
  if (src.close) src.close();
  const imageData = ctx.getImageData(0, 0, cw, ch);
  return { image: { data: imageData.data, width: cw, height: ch, channels: 4 }, imageData, downscale: f };
}

export class Scorer {
  async init(onProgress = () => {}) {
    const [config, stats, alignment] = await Promise.all(['config.json', 'stats.json', 'alignment.json'].map(loadJSON));
    Object.assign(this, { config, stats, alignment });

    ort.env.wasm.numThreads = 1; // single threaded: deterministic and no COOP/COEP needed
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = url('vendor/ort/');
    ort.env.logLevel = 'error';

    onProgress({ stage: 'model', loaded: 0, total: config.model.bytes });
    const bytes = await fetchWithProgress(config.model.file, config.model.bytes, (loaded, total) => onProgress({ stage: 'model', loaded, total }));
    onProgress({ stage: 'runtime' });
    this.session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    this.inputName = this.session.inputNames[0];
    this.outputName = this.session.outputNames[0];

    onProgress({ stage: 'landmarker' });
    const fileset = await FilesetResolver.forVisionTasks(url('vendor/mediapipe/wasm'));
    this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: url('models/face_landmarker.task'), delegate: 'CPU' },
      runningMode: 'IMAGE',
      numFaces: 3,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    onProgress({ stage: 'ready' });
    return this;
  }

  // Run the network on a 350x350 aligned crop. Returns the raw float32 output.
  async scoreCrop(crop) {
    const x = toTensor(crop, this.config.model.preprocess);
    const out = await this.session.run({ [this.inputName]: new ort.Tensor('float32', x, [1, 3, 224, 224]) });
    return out[this.outputName].data[0];
  }

  // Full pipeline for one image blob. Never throws for "bad photo" cases; returns status instead.
  async process(blob) {
    const buf = await blob.arrayBuffer();
    const hash = await sha256Hex(buf);
    let dec;
    try {
      dec = await decode(blob, this.config.max_image_pixels);
    } catch (e) {
      return { hash, status: 'error', message: e.message || 'Could not read this image' };
    }
    const { image, imageData } = dec;
    const res = this.landmarker.detect(imageData);
    const nFaces = res.faceLandmarks.length;
    if (nFaces === 0) return { hash, status: 'noface', message: 'No face found' };
    if (nFaces > 1) return { hash, status: 'multiface', faces: nFaces, message: `${nFaces} faces found, need exactly one` };

    const pts = res.faceLandmarks[0].map((p) => [p.x * image.width, p.y * image.height]);
    const { crop, T } = align(image, pts, this.alignment);
    const raw = await this.scoreCrop(crop);

    // ---- quality checks ----
    const q = this.config.quality;
    const flags = [];
    const add = (kind, label) => flags.push({ kind, label, exclude: !!q.exclude[kind] });

    const ref = this.alignment.pose_reference_deg;
    let pose = null;
    const mats = res.facialTransformationMatrixes;
    if (mats && mats[0]) {
      const e = eulerDeg(mats[0].data);
      pose = { yaw: e.yaw - ref.yaw, pitch: e.pitch - ref.pitch, roll: e.roll - ref.roll };
      const parts = [];
      if (Math.abs(pose.yaw) > q.max_pose_deg) parts.push(`turned ${Math.round(Math.abs(pose.yaw))}°`);
      if (Math.abs(pose.pitch) > q.max_pose_deg) parts.push(`tilted up/down ${Math.round(Math.abs(pose.pitch))}°`);
      if (Math.abs(pose.roll) > q.max_pose_deg) parts.push(`tilted sideways ${Math.round(Math.abs(pose.roll))}°`);
      if (parts.length) add('pose', 'Head ' + parts.join(', '));
    }
    const bs = {};
    for (const c of res.faceBlendshapes?.[0]?.categories || []) bs[c.categoryName] = c.score;
    const smile = Math.max(bs.mouthSmileLeft || 0, bs.mouthSmileRight || 0);
    const jaw = bs.jawOpen || 0;
    if (smile > q.smile_threshold) add('expression', 'Smiling (use a neutral face)');
    if (jaw > q.jaw_open_threshold) add('expression', 'Mouth open');

    const [l, r] = eyeCenters(pts);
    const eyeDist = Math.hypot(r[0] - l[0], r[1] - l[1]) / dec.downscale;
    if (eyeDist < q.min_eye_distance_px) add('small_face', `Face too small (${Math.round(eyeDist)} px between eyes)`);

    const gray = grayOf(crop);
    const sharpness = laplacianVar(gray, crop.width, crop.height);
    const brightness = faceBrightness(gray, crop.width);
    if (sharpness < q.min_sharpness) add('blur', 'Blurry or low resolution');
    if (brightness < q.min_brightness) add('lighting', 'Too dark');
    if (brightness > q.max_brightness) add('lighting', 'Too bright');

    return {
      hash, status: 'ok', raw, crop, flags,
      excluded: flags.some((f) => f.exclude),
      metrics: { pose, smile, jaw, eyeDist, sharpness, brightness, scale: T.s, width: image.width, height: image.height },
    };
  }
}
