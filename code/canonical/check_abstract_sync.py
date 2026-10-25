#!/usr/bin/env python3
"""Programmatic consistency check: PDF(main.tex) abstract vs
SUBMISSION_ABSTRACT.md blockquote. Normalizes LaTeX/markdown noise and
compares token streams; exits 1 on divergence."""
import re, sys
from pathlib import Path
R = Path('__SRTS_ROOT__/calattn-repo')
tex = (R/'www2027/paper/main.tex').read_text()
a = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S).group(1)
md = (R/'www2027/docs/SUBMISSION_ABSTRACT.md').read_text()
b = '\n'.join(l[2:] for l in md.splitlines() if l.startswith('> '))
def norm(x):
    x = re.sub(r'\\(textbf|emph)\{([^}]*)\}', r'\2', x)
    x = x.replace('\\%','%').replace('~',' ').replace('---','--').replace('–','--')
    x = re.sub(r'\$([^$]*)\$', lambda m: m.group(1).replace('\\le','<=').replace('{=}','=').replace('\\dots','...').replace('\\in',' in ').replace(',\\dots,',',...,'), x)
    x = re.sub(r'\s+',' ', x).strip().lower()
    x = re.sub(r'[{}\\]','', x)
    return x
na, nb = norm(a), norm(b)
if na == nb:
    print('ABSTRACTS IN SYNC (normalized,', len(na), 'chars)')
    sys.exit(0)
import difflib
for d in difflib.unified_diff(na.split(' . '), nb.split(' . '), lineterm=''):
    print(d)
# token-level first divergence
ta, tb = na.split(), nb.split()
for i,(x,y) in enumerate(zip(ta,tb)):
    if x!=y:
        print('FIRST DIVERGENCE @token',i,':',ta[max(0,i-5):i+5],'VS',tb[max(0,i-5):i+5]); break
sys.exit(1)
