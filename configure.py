#!/usr/bin/env python3
"""Rewrite __SRTS_ROOT__ in code/ to an absolute work root and link the
layout the scripts expect (results, anchor cells, paper outputs)."""
import sys, pathlib
root = sys.argv[1] if len(sys.argv) > 1 else None
if not root or not root.startswith('/'):
    sys.exit('usage: python configure.py /abs/path/to/workdir')
n = 0
for f in pathlib.Path('code').rglob('*.py'):
    t = f.read_text()
    if '__SRTS_ROOT__' in t:
        f.write_text(t.replace('__SRTS_ROOT__', root)); n += 1
pkg = pathlib.Path('.').resolve(); r = pathlib.Path(root)
links = [(r / 'calattn-repo/www2027/reviewed_artifact/srts_bce_artifact/data/cells', pkg / 'data/c100_cells'),
         (r / 'calattn-repo/results/www2027', pkg / 'results'),
         (r / 'calattn-repo/www2027/paper', pkg / 'paper')]
for link, target in links:
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(target)
import gzip, shutil
k = 0
for g in pathlib.Path('results').rglob('*.jsonl.gz'):
    out = g.with_suffix('')
    if not out.exists():
        with gzip.open(g, 'rb') as fi, open(out, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        k += 1
(pkg / 'configured_root.txt').write_text(root)
print(f'configured {n} files; decompressed {k} row files; linked',
      [str(l) for l, _ in links])
