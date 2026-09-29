// Score conversions and aggregation. Pure functions, driven by config.json and stats.json.

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// Empirical percentile of `raw` among SCUT TRAIN labels, interpolated in the 1..99 table.
export function percentile(raw, table) {
  const ps = Object.keys(table).map(Number).sort((a, b) => a - b);
  const vs = ps.map((p) => table[String(p)]);
  if (raw < vs[0]) return { value: ps[0], label: `<${ps[0]}` };
  if (raw >= vs[vs.length - 1]) return { value: ps[ps.length - 1], label: `>${ps[ps.length - 1]}` };
  for (let i = vs.length - 2; i >= 0; i--) {
    if (raw >= vs[i]) {
      const span = vs[i + 1] - vs[i];
      const v = span > 0 ? ps[i] + ((raw - vs[i]) / span) * (ps[i + 1] - ps[i]) : ps[i];
      return { value: v, label: String(Math.round(v)) };
    }
  }
  return { value: ps[0], label: String(ps[0]) };
}

export function convert(raw, gender, stats, config) {
  const g = stats[gender];
  const z = (raw - g.mean) / g.std;
  const c = config.conversions;
  const psl = clamp(c.psl.base + c.psl.per_z * z, c.psl.min, c.psl.max);
  const ten = clamp(c.ten.base + c.ten.per_z * z, c.ten.min, c.ten.max);
  return { raw, z, psl, ten, percentile: percentile(raw, g.percentiles) };
}

// photos: [{ raw, excluded, hash }] (only successfully scored photos).
// The same file added twice (same SHA-256) is counted once.
export function aggregate(photos, includeFlagged, config) {
  const seen = new Set();
  const unique = photos.filter((p) => {
    if (p.hash && seen.has(p.hash)) return false;
    if (p.hash) seen.add(p.hash);
    return true;
  });
  const used = unique.filter((p) => includeFlagged || !p.excluded);
  if (!used.length) return null;
  const raws = used.map((p) => p.raw);
  const mean = raws.reduce((a, b) => a + b, 0) / raws.length;
  const spread = Math.max(...raws) - Math.min(...raws);
  return { raw: mean, n: used.length, flaggedOut: unique.length - used.length, duplicates: photos.length - unique.length, spread, disagree: used.length > 1 && spread > config.spread_warning };
}

export function ordinal(n) {
  const s = ['th', 'st', 'nd', 'rd'], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}
