// Service worker: offline support. Small app files use stale while revalidate
// (edits such as config.json show up on the next load); the big immutable
// assets (models, wasm) are cache first. Bump VERSION to drop old caches.
const VERSION = 'psl-v1';
const SHELL = [
  './', 'index.html', 'css/app.css', 'js/app.js', 'js/scorer.js', 'js/pipeline.js', 'js/scoring.js',
  'config.json', 'stats.json', 'alignment.json', 'manifest.webmanifest',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png',
  'vendor/ort/ort.wasm.min.mjs', 'vendor/ort/ort-wasm-simd-threaded.mjs',
  'vendor/mediapipe/vision_bundle.mjs', 'vendor/mediapipe/wasm/vision_wasm_internal.js',
];
const BIG = [
  'models/scut_resnet18.onnx', 'models/face_landmarker.task',
  'vendor/ort/ort-wasm-simd-threaded.wasm', 'vendor/mediapipe/wasm/vision_wasm_internal.wasm',
];
const isBig = (url) => /\.(onnx|task|wasm)$/.test(new URL(url).pathname);

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// The page asks for the big files once it has loaded them (usually straight from the HTTP cache).
self.addEventListener('message', (e) => {
  if (e.data?.type !== 'precache') return;
  e.waitUntil(caches.open(VERSION).then(async (c) => {
    for (const u of BIG) {
      const req = new Request(new URL(u, self.registration.scope).href);
      if (!(await c.match(req))) {
        try { await c.add(req); } catch (err) { /* retried next visit */ }
      }
    }
  }));
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin || req.headers.has('range')) return;
  e.respondWith(caches.open(VERSION).then(async (cache) => {
    const hit = await cache.match(req, { ignoreSearch: true });
    if (hit && isBig(req.url)) return hit;
    const net = fetch(req).then((res) => {
      if (res.ok && res.status === 200 && res.type === 'basic') cache.put(req, res.clone());
      return res;
    });
    if (hit) { e.waitUntil(net.catch(() => {})); return hit; }
    return net.catch(() => cache.match('index.html'));
  }));
});
