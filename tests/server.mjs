// Tiny static server for the tests: / -> web/, /fixtures/ -> tests/out/fixtures, /scut/ -> training/data/images
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MOUNTS = [
  ['/fixtures/', path.join(root, 'tests', 'out', 'fixtures')],
  ['/scut/', path.join(root, 'training', 'data', 'images')],
  ['/', path.join(root, 'web')],
];
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.webmanifest': 'application/manifest+json', '.wasm': 'application/wasm',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.onnx': 'application/octet-stream',
  '.task': 'application/octet-stream', '.map': 'application/json',
};

export function serve(port = 0) {
  const server = http.createServer((req, res) => {
    const u = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    for (const [prefix, dir] of MOUNTS) {
      if (!u.startsWith(prefix)) continue;
      let p = path.join(dir, u.slice(prefix.length));
      if (!p.startsWith(dir)) break;
      if (fs.existsSync(p) && fs.statSync(p).isDirectory()) p = path.join(p, 'index.html');
      if (!fs.existsSync(p)) break;
      const stat = fs.statSync(p);
      res.writeHead(200, { 'content-type': TYPES[path.extname(p)] || 'application/octet-stream', 'content-length': stat.size, 'cache-control': 'no-cache' });
      fs.createReadStream(p).pipe(res);
      return;
    }
    res.writeHead(404); res.end('not found');
  });
  return new Promise((resolve) => server.listen(port, '127.0.0.1', () => resolve(server)));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const s = await serve(Number(process.argv[2] || 8080));
  console.log('serving on', s.address().port);
}
