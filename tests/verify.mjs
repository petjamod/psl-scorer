// Phase 4 verification: runs the REAL web app (served locally) in headless Chromium.
// Prerequisite: python training/make_fixtures.py (writes tests/out/fixtures).
// Output: tests/out/results.json and screenshots; training/write_validation.py turns them into VALIDATION.md.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { serve } from './server.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, 'out');
const FX = path.join(OUT, 'fixtures');
const e2e = JSON.parse(fs.readFileSync(path.join(FX, 'e2e.json'), 'utf8'));
const qman = JSON.parse(fs.readFileSync(path.join(FX, 'quality', 'manifest.json'), 'utf8'));
const results = { when: new Date().toISOString(), model: e2e.model, variant: e2e.variant };

function pearson(a, b) {
  const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return sab / Math.sqrt(saa * sbb);
}
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

const server = await serve(0);
const base = `http://127.0.0.1:${server.address().port}/`;
const browser = await chromium.launch();

async function openApp(ctxOpts = {}) {
  const ctx = await browser.newContext(ctxOpts);
  const page = await ctx.newPage();
  page.on('pageerror', (e) => log('[pageerror]', e.message));
  await page.goto(base);
  await page.evaluate(() => window.__psl.ready.then(() => true));
  await page.evaluate(() => {
    window.__score = async (url) => {
      const b = await (await fetch(url)).blob();
      const t = performance.now();
      const r = await window.__psl.scorer.process(b);
      return {
        ms: performance.now() - t, status: r.status, raw: r.raw,
        bits: r.raw === undefined ? null : new Uint32Array(new Float32Array([r.raw]).buffer)[0],
        flags: (r.flags || []).map((f) => f.kind), labels: (r.flags || []).map((f) => f.label),
        excluded: r.excluded, metrics: r.metrics, hash: r.hash, message: r.message,
      };
    };
  });
  return { ctx, page };
}

const { ctx, page } = await openApp();
log('app ready');

// 1. End to end accuracy on 300 SCUT test images (alignment included).
{
  const js = [], ms = [], eyeErr = [];
  for (const [k, it] of e2e.items.entries()) {
    const r = await page.evaluate((u) => window.__score(u), `/scut/${it.png}`);
    if (r.status !== 'ok') { log('not ok', it.name, r.status); js.push(null); continue; }
    js.push(r.raw); ms.push(r.ms);
    const [l, rr] = r.metrics.eyes, [pl, pr] = it.py_eyes;
    eyeErr.push((Math.hypot(l[0] - pl[0], l[1] - pl[1]) + Math.hypot(rr[0] - pr[0], rr[1] - pr[1])) / 2);
    if (k % 50 === 0) log('e2e', k);
  }
  const ok = e2e.items.map((it, i) => ({ it, v: js[i] })).filter((x) => x.v !== null);
  const labels = ok.map((x) => x.it.score);
  const jsRaw = ok.map((x) => x.v);
  const pyRaw = ok.map((x) => x.it.py_raw_aligned);
  const diffs = ok.map((x) => Math.abs(x.v - x.it.py_raw_aligned));
  results.e2e = {
    n: e2e.items.length, scored: ok.length,
    pearson_js: pearson(jsRaw, labels),
    pearson_python_same_images: pearson(pyRaw, labels),
    pearson_python_unaligned_same_images: pearson(ok.map((x) => x.it.py_raw_unaligned), labels),
    mae_js: mean(ok.map((x) => Math.abs(x.v - x.it.score))),
    js_vs_python_abs_diff_mean: mean(diffs), js_vs_python_abs_diff_max: Math.max(...diffs),
    eye_center_px_diff_mean: mean(eyeErr), eye_center_px_diff_max: Math.max(...eyeErr),
    ms_per_image_mean: mean(ms),
    raws: e2e.items.map((it, i) => [it.name, js[i]]),
  };
  log('e2e', JSON.stringify({ ...results.e2e, raws: undefined }));
}

// 2. Preprocessing parity: JS tensor from the raw PNG vs Python tensor (no alignment involved).
{
  const rows = [];
  for (const it of e2e.items.slice(0, 20)) {
    rows.push(await page.evaluate(async ({ png, variant }) => {
      const { decode } = await import('/js/scorer.js');
      const P = await import('/js/pipeline.js');
      const b = await (await fetch(`/scut/${png}`)).blob();
      const dec = await decode(b);
      const t = P.toTensor(P.toRGB(dec.image), variant);
      const py = new Float32Array(await (await fetch(`/fixtures/tensors/${png}.f32`)).arrayBuffer());
      let maxd = 0, nd = 0;
      for (let i = 0; i < t.length; i++) { const d = Math.abs(t[i] - py[i]); if (d > 0) nd++; if (d > maxd) maxd = d; }
      const ort = window.__psl.scorer;
      const raw = await ort.scoreCrop(P.toRGB(dec.image));
      return { maxd, nd, raw };
    }, { png: it.png, variant: e2e.variant }));
  }
  results.tensor_parity = {
    images: rows.length,
    tensor_max_abs_diff: Math.max(...rows.map((r) => r.maxd)),
    tensor_values_differing: rows.reduce((s, r) => s + r.nd, 0),
    onnx_web_vs_python_ort_max_abs_diff: Math.max(...rows.map((r, i) => Math.abs(r.raw - e2e.items[i].py_raw_unaligned))),
  };
  log('parity', JSON.stringify(results.tensor_parity));
}

// 3. Determinism: one image scored 20 times, plus once more in a fresh browser context.
{
  const url = `/scut/${e2e.items[0].png}`;
  const runs = [];
  for (let i = 0; i < 20; i++) runs.push(await page.evaluate((u) => window.__score(u), url));
  const fresh = await openApp();
  const again = await fresh.page.evaluate((u) => window.__score(u), url);
  await fresh.ctx.close();
  const bits = runs.map((r) => r.bits);
  results.determinism = {
    runs: runs.length, unique_outputs: new Set(bits).size, raw: runs[0].raw, float32_bits_hex: bits[0].toString(16),
    fresh_context_identical: again.bits === bits[0], bit_identical: new Set(bits).size === 1 && again.bits === bits[0],
  };
  log('determinism', JSON.stringify(results.determinism));
}

// 4. Robustness to JPEG re-encoding (50 images, quality 85 and 70).
{
  const pngRaw = Object.fromEntries(results.e2e.raws);
  const out = {};
  for (const qv of [85, 70]) {
    const d = [];
    for (const it of e2e.items.slice(0, 50)) {
      const r = await page.evaluate((u) => window.__score(u), `/fixtures/jpeg${qv}/${it.png.replace('.png', '.jpg')}`);
      if (r.status === 'ok' && pngRaw[it.name] != null) d.push(r.raw - pngRaw[it.name]);
    }
    out[`q${qv}`] = { n: d.length, mean_abs_change: mean(d.map(Math.abs)), max_abs_change: Math.max(...d.map(Math.abs)), mean_signed_change: mean(d) };
  }
  results.jpeg = out;
  log('jpeg', JSON.stringify(out));
}

// 5. Quality gate sanity.
{
  const out = {};
  for (const [name, spec] of Object.entries(qman)) {
    const file = fs.existsSync(path.join(FX, 'quality', `${name}.png`)) ? `${name}.png` : `${name}.jpg`;
    const r = await page.evaluate((u) => window.__score(u), `/fixtures/quality/${file}`);
    let pass;
    if (spec.expect_status) pass = r.status === spec.expect_status;
    else pass = r.status === 'ok' && spec.expect.every((k) => r.flags.includes(k));
    if (name === 'neutral') pass = r.status === 'ok' && r.flags.length === 0;
    out[name] = { pass, status: r.status, flags: r.labels, raw: r.raw, message: r.message, pose: r.metrics?.pose, smile: r.metrics?.smile, spec };
  }
  const up = await page.evaluate((u) => window.__score(u), '/fixtures/quality/exif_upright.jpg');
  out.exif_rot6.upright_raw = up.raw;
  out.exif_rot6.abs_diff_vs_upright = Math.abs(out.exif_rot6.raw - up.raw);
  out.exif_rot6.pass = out.exif_rot6.pass && out.exif_rot6.abs_diff_vs_upright < 0.05;
  out.big.abs_diff_vs_python = Math.abs(out.big.raw - qman.big.py_raw);
  out.big.abs_diff_vs_original_350 = Math.abs(out.big.raw - qman.big.raw_of_original_350);
  out.big.pass = out.big.pass && out.big.abs_diff_vs_python < 0.01;
  results.quality = out;
  log('quality', JSON.stringify(Object.fromEntries(Object.entries(out).map(([k, v]) => [k, [v.pass, v.status, v.flags]]))));
}
await ctx.close();

// 6. Mobile layout at 390x844 through the real UI (file input), plus "seen before".
{
  const m = await openApp({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const p = m.page;
  await p.click('.seg button[data-gender="male"]');
  const files = ['neutral.png', 'smiling.png', 'rotated30.png', 'two_faces.png'].map((f) => path.join(FX, 'quality', f));
  await p.setInputFiles('#pick', files);
  await p.waitForFunction(() => window.__psl.state.photos.length === 4 && window.__psl.state.photos.every((x) => !['pending', 'working'].includes(x.status)), null, { timeout: 60000 });
  await p.screenshot({ path: path.join(OUT, 'mobile_390x844.png') });
  await p.screenshot({ path: path.join(OUT, 'mobile_390_full.png'), fullPage: true });
  const layout = await p.evaluate(() => {
    const W = window.innerWidth;
    const offenders = [];
    for (const e of document.querySelectorAll('body *')) {
      const r = e.getBoundingClientRect();
      if (r.width && (r.right > W + 0.5 || r.left < -0.5)) offenders.push(`${e.tagName}.${e.className} ${Math.round(r.left)}..${Math.round(r.right)}`);
    }
    return { innerWidth: W, scrollWidth: document.documentElement.scrollWidth, offenders: offenders.slice(0, 10) };
  });
  const ui = await p.evaluate(() => ({
    psl: document.getElementById('rPsl').textContent, ten: document.getElementById('rTen').textContent,
    raw: document.getElementById('rRaw').textContent, pct: document.getElementById('rPct').textContent,
    meta: document.getElementById('rMeta').textContent,
    cards: [...document.querySelectorAll('.photo')].map((c) => ({ cls: c.className, text: c.innerText.replace(/\s+/g, ' ') })),
  }));
  // Upload the neutral photo again: must be marked "seen before" with the identical score.
  await p.setInputFiles('#pick', [files[0]]);
  await p.waitForFunction(() => window.__psl.state.photos.length === 5 && window.__psl.state.photos.every((x) => !['pending', 'working'].includes(x.status)));
  const seen = await p.evaluate(() => {
    const ph = window.__psl.state.photos;
    return { seen: !!ph[4].seen, same: ph[4].raw === ph[0].raw, badge: !!document.querySelectorAll('.photo')[4].querySelector('.badge.seen') };
  });
  // Toggle "include flagged" and check the average changes accordingly.
  await p.click('label.switch');
  const withFlagged = await p.evaluate(() => document.getElementById('rMeta').textContent);
  results.mobile = {
    ...layout, overflow: layout.scrollWidth > layout.innerWidth || layout.offenders.length > 0, ui, seen_before: seen, meta_with_flagged: withFlagged,
  };
  log('mobile', JSON.stringify({ ...results.mobile, ui: { ...ui, cards: ui.cards.length } }));

  // 7. Offline after first load (PWA): wait for the service worker cache, go offline, reload.
  await p.evaluate(async () => {
    const reg = await navigator.serviceWorker.ready;
    reg.active.postMessage({ type: 'precache' });
    for (let i = 0; i < 100; i++) {
      const c = await caches.open((await caches.keys())[0]);
      const keys = (await c.keys()).map((r) => r.url);
      if (['scut_resnet18.onnx', 'face_landmarker.task', 'ort-wasm-simd-threaded.wasm', 'vision_wasm_internal.wasm'].every((n) => keys.some((k) => k.endsWith(n)))) return keys.length;
      await new Promise((r) => setTimeout(r, 200));
    }
    return -1;
  });
  await m.ctx.setOffline(true);
  let offline = false;
  try {
    await p.reload();
    offline = await p.evaluate(() => window.__psl.ready.then(() => true));
    const r = await p.evaluate(async () => (await caches.keys()).length);
    results.offline = { app_ready_offline: offline, caches: r };
  } catch (e) {
    results.offline = { app_ready_offline: false, error: String(e.message || e) };
  }
  log('offline', JSON.stringify(results.offline));
  await m.ctx.close();
}

fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify(results, null, 1));
await browser.close();
server.close();
log('done');
