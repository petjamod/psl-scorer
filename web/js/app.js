import { Scorer } from './scorer.js';
import { convert, aggregate, ordinal } from './scoring.js';

const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage full or blocked */ } },
};

const state = {
  gender: store.get('psl:gender'),
  includeFlagged: store.get('psl:includeFlagged') === '1',
  photos: [], // { id, name, status, raw, flags, excluded, seen, crop, message }
  nextId: 1,
};

const scorer = new Scorer();
const ready = scorer.init(onLoadProgress).then(() => {
  $('loader').hidden = true;
  $('accLine').textContent = `Pearson correlation ${scorer.config.model.test_pearson.toFixed(2)} with held out human ratings on 2,200 test faces`;
  if (navigator.serviceWorker?.controller) navigator.serviceWorker.controller.postMessage({ type: 'precache' });
  else navigator.serviceWorker?.ready.then((r) => r.active?.postMessage({ type: 'precache' }));
  return scorer;
}).catch((e) => {
  console.error(e);
  $('loader').classList.add('error');
  $('loaderText').textContent = 'Could not load the model: ' + (e.message || e);
  $('loaderHint').textContent = 'Check your connection and reload.';
  throw e;
});

function onLoadProgress(p) {
  const text = $('loaderText'), pct = $('loaderPct'), bar = $('loaderBar');
  if (p.stage === 'model') {
    const f = p.total ? p.loaded / p.total : 0;
    text.textContent = 'Downloading model';
    pct.textContent = `${(p.loaded / 1e6).toFixed(1)} / ${(p.total / 1e6).toFixed(1)} MB`;
    bar.style.width = `${Math.round(f * 90)}%`;
  } else if (p.stage === 'runtime') {
    text.textContent = 'Starting the runtime'; pct.textContent = ''; bar.style.width = '93%';
  } else if (p.stage === 'landmarker') {
    text.textContent = 'Loading the face detector'; bar.style.width = '97%';
  } else if (p.stage === 'ready') {
    bar.style.width = '100%';
  }
}

// ---- result cache: SHA-256 of the file -> raw score ----
function cacheKey(hash) { return `psl:r:${scorer.config.model.id}:${hash}`; }
function cacheGet(hash) {
  const v = store.get(cacheKey(hash));
  if (!v) return null;
  try { return JSON.parse(v); } catch { return null; }
}
function cachePut(hash, raw) { store.set(cacheKey(hash), JSON.stringify({ raw, t: Date.now() })); }

// ---- input ----
async function addFiles(fileList) {
  const files = [...fileList].filter((f) => f.type.startsWith('image/') || /\.(jpe?g|png|webp|heic|heif|avif|gif|bmp)$/i.test(f.name));
  const room = scorer.config ? scorer.config.max_photos - state.photos.length : 10 - state.photos.length;
  if (files.length > room) alert(`Up to 10 photos. Only the first ${Math.max(0, room)} of these were added.`);
  const todo = files.slice(0, Math.max(0, room)).map((f) => {
    const p = { id: state.nextId++, name: f.name || 'camera photo', status: 'pending', file: f };
    state.photos.push(p);
    return p;
  });
  render();
  await ready;
  for (const p of todo) {
    p.status = 'working'; render();
    await processOne(p);
    render();
  }
}

async function processOne(p) {
  try {
    const r = await scorer.process(p.file);
    Object.assign(p, r);
    if (r.status === 'ok') {
      const prev = cacheGet(r.hash);
      if (prev) {
        p.seen = true;
        if (prev.raw !== r.raw) console.warn('Score differs from the cached one', prev.raw, r.raw);
        p.raw = prev.raw; // same photo, same score
      } else {
        cachePut(r.hash, r.raw);
      }
    }
  } catch (e) {
    console.error(e);
    p.status = 'error';
    p.message = e.message || 'Could not process this photo';
  }
  delete p.file;
}

// ---- rendering ----
function drawCrop(canvas, crop) {
  canvas.width = crop.width; canvas.height = crop.height;
  const ctx = canvas.getContext('2d');
  const id = ctx.createImageData(crop.width, crop.height);
  for (let i = 0, j = 0; i < crop.data.length; i += 3, j += 4) {
    id.data[j] = crop.data[i]; id.data[j + 1] = crop.data[i + 1]; id.data[j + 2] = crop.data[i + 2]; id.data[j + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

const cardCache = new Map();
function photoCard(p) {
  let card = cardCache.get(p.id);
  if (!card) {
    card = el('article', 'photo');
    card.dataset.id = p.id;
    cardCache.set(p.id, card);
  }
  const key = [p.status, p.raw, p.excluded, p.seen, state.gender, state.includeFlagged].join('|');
  if (card.dataset.key === key) return card;
  card.dataset.key = key;
  card.replaceChildren();
  card.className = 'photo';
  const d = scorer.config?.conversions;

  if (p.status === 'ok') {
    const cv = el('canvas');
    cv.setAttribute('aria-label', 'Aligned 350 by 350 crop the model sees');
    drawCrop(cv, p.crop);
    cv.addEventListener('click', () => card.classList.toggle('zoom'));
    card.append(cv);
  } else {
    card.append(el('div', 'ph'));
  }
  const body = el('div', 'body');
  const line1 = el('div', 'line1');
  body.append(line1);

  if (p.status === 'pending' || p.status === 'working') {
    const s = el('span', 'score');
    s.style.fontSize = '15px';
    s.innerHTML = '<span class="spinner"></span>';
    s.append(p.status === 'pending' ? 'Waiting' : 'Analyzing');
    line1.append(s);
  } else if (p.status === 'ok') {
    card.classList.toggle('excluded', p.excluded && !state.includeFlagged);
    line1.append(el('span', 'score', p.raw.toFixed(d.raw_decimals)), el('span', 'of', 'raw / 5'));
    if (p.seen) line1.append(el('span', 'badge seen', 'seen before'));
    if (state.gender) {
      const c = convert(p.raw, state.gender, scorer.stats, scorer.config);
      body.append(el('div', 'conv', `PSL ${c.psl.toFixed(d.psl.decimals)} · ${c.ten.toFixed(d.ten.decimals)}/10`));
    }
    const flags = el('div', 'flags');
    for (const f of p.flags) flags.append(el('span', f.exclude ? 'flag ex' : 'flag', f.label));
    if (!p.flags.length) flags.append(el('span', 'flag ok', 'Checks passed'));
    else if (p.excluded) flags.append(el('span', 'flag ex', state.includeFlagged ? 'Included anyway' : 'Not averaged'));
    body.append(flags);
  } else {
    card.classList.add('failed', 'excluded');
    line1.append(el('span', 'score', p.status === 'multiface' ? 'Skipped' : 'No score'));
    const flags = el('div', 'flags');
    flags.append(el('span', 'flag ex', p.message || 'Could not process'));
    body.append(flags);
  }
  body.append(el('div', 'name', p.name));
  card.append(body);
  return card;
}

function render() {
  const list = $('photos');
  const cards = state.photos.map(photoCard);
  list.replaceChildren(...cards);
  $('clear').hidden = state.photos.length === 0;
  for (const b of document.querySelectorAll('.seg button')) b.setAttribute('aria-checked', String(b.dataset.gender === state.gender));
  $('includeFlagged').checked = state.includeFlagged;
  renderResult();
}

function renderResult() {
  const scored = state.photos.filter((p) => p.status === 'ok');
  const box = $('result');
  if (!scored.length || !scorer.config) { box.hidden = true; return; }
  box.hidden = false;
  const agg = aggregate(scored, state.includeFlagged, scorer.config);
  const d = scorer.config.conversions;
  $('rNone').hidden = !!agg;
  $('rDisagree').hidden = !(agg && agg.disagree);
  $('rNoGender').hidden = !(agg && !state.gender);
  if (!agg) {
    for (const id of ['rPsl', 'rTen', 'rRaw', 'rPct']) $(id).textContent = '…';
    $('rMeta').textContent = '';
    return;
  }
  $('rRaw').textContent = agg.raw.toFixed(d.raw_decimals);
  if (state.gender) {
    const c = convert(agg.raw, state.gender, scorer.stats, scorer.config);
    $('rPsl').textContent = c.psl.toFixed(d.psl.decimals);
    $('rTen').textContent = c.ten.toFixed(d.ten.decimals);
    const pl = c.percentile.label;
    $('rPct').textContent = /^\d+$/.test(pl) ? ordinal(Number(pl)) : pl;
  } else {
    for (const id of ['rPsl', 'rTen', 'rPct']) $(id).textContent = '…';
  }
  const skipped = scored.length - agg.n;
  const g = state.gender ? ` · vs SCUT ${state.gender === 'male' ? 'men' : 'women'}` : '';
  $('rMeta').textContent = `Averaged over ${agg.n} photo${agg.n > 1 ? 's' : ''}` +
    (agg.n > 1 ? ` · spread ${agg.spread.toFixed(2)}` : '') +
    (skipped ? ` · ${skipped} flagged not averaged` : '') + g;
}

// ---- wiring ----
for (const b of document.querySelectorAll('.seg button')) {
  b.addEventListener('click', () => { state.gender = b.dataset.gender; store.set('psl:gender', state.gender); render(); });
}
$('includeFlagged').addEventListener('change', (e) => {
  state.includeFlagged = e.target.checked; store.set('psl:includeFlagged', state.includeFlagged ? '1' : '0'); render();
});
for (const id of ['pick', 'camera']) {
  $(id).addEventListener('change', (e) => { const f = e.target.files; if (f?.length) addFiles(f); e.target.value = ''; });
}
$('clear').addEventListener('click', () => { state.photos = []; cardCache.clear(); render(); });
render();

if ('serviceWorker' in navigator && location.protocol !== 'file:') {
  navigator.serviceWorker.register('sw.js').catch((e) => console.warn('SW registration failed', e));
}

// Hook for the Playwright verification suite (tests/): runs the exact same pipeline.
window.__psl = { ready, scorer, addFiles, state };
