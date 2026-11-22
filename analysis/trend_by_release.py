"""Score vs model release date within single leaderboard snapshots.

Usage: python3 -I trend_by_release.py <data_dir> <bullshitbench_launch_csv>
Each board is one frozen snapshot (same test, same judge), so models are comparable inside a board.
"""
import csv, glob, os, re, sys, statistics as st
from collections import defaultdict

data_dir, bsb_launch = sys.argv[1], sys.argv[2]


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import identity  # shared model identity: canonical name plus release tag

rows, rel, _conflicts = identity.load(data_dir, bsb_launch)


def key_of(r):
    return r['ident']


SERIES = [
    ('aa_v11', 'net_index', 1, 'AA-Omniscience v1.1 net index (higher better)'),
    ('aa_v11', 'accuracy', 1, 'AA-Omniscience v1.1 accuracy % (higher better)'),
    ('aa_v11', 'hallucination_rate', -1, 'AA-Omniscience v1.1 hallucination rate % (lower better)'),
    ('aa_v11', 'attempt_rate', 0, 'AA-Omniscience v1.1 attempt rate %'),
    ('aa_v1', 'net_index', 1, 'AA-Omniscience v1 net index (higher better)'),
    ('aa_v1', 'accuracy', 1, 'AA-Omniscience v1 accuracy % (higher better)'),
    ('aa_v1', 'hallucination_rate', -1, 'AA-Omniscience v1 hallucination rate % (lower better)'),
    ('aa_v1', 'attempt_rate', 0, 'AA-Omniscience v1 attempt rate %'),
    ('bullshitbench_v2', 'green_rate', 1, 'BullshitBench pushback rate 0-1 (higher better)'),
    ('bullshitbench_v2', 'red_rate', -1, 'BullshitBench accepted-nonsense rate 0-1 (lower better)'),
    ('simpleqa_verified_epoch', 'accuracy', 1, 'SimpleQA Verified (Epoch, forced guess) accuracy 0-1 (higher better)'),
    ('facts_v2', 'grounding_score', 1, 'FACTS grounding % (higher better)'),
    ('facts_v2', 'parametric_score', 1, 'FACTS parametric % (higher better)'),
    ('vectara_new', 'hallucination_rate', -1, 'Vectara new board hallucination rate % (lower better)'),
    ('vectara_old', 'hallucination_rate', -1, 'Vectara old board hallucination rate % (lower better)'),
]
BIG = {'openai', 'anthropic', 'google', 'xai', 'meta'}


def bucket(d):
    y, m = int(d[:4]), int(d[5:7])
    if y <= 2023:
        return '2023-'
    return f'{y}H{1 if m <= 6 else 2}'


def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        for t in range(i, j + 1):
            r[o[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den if den else float('nan')


for board, metric, sign, label in SERIES:
    vals, vend = defaultdict(list), {}
    for r in rows:
        if r['board'] == board and r['metric'] == metric:
            try:
                vals[r['ident']].append(float(r['value']))
            except ValueError:
                pass
            vend[r['ident']] = r['vendor'].lower()
    pts = [(rel[k], st.mean(v), vend[k]) for k, v in vals.items() if k in rel]
    print(f'\n== {label}   models with a release date: {len(pts)} of {len(vals)}')
    for name, sub in (('all', pts), ('big 5 labs', [p for p in pts if any(b in p[2] for b in BIG)])):
        if len(sub) < 8:
            continue
        days = [int(p[0][:4]) * 372 + int(p[0][5:7]) * 31 + int(p[0][8:]) for p in sub]
        rho = spearman(days, [p[1] for p in sub])
        by = defaultdict(list)
        for d, v, _ in sub:
            by[bucket(d)].append(v)
        best = max if sign >= 0 else min
        cells = [f"{b}: med {st.median(v):.2f} best {best(v):.2f} n={len(v)}" for b, v in sorted(by.items())]
        print(f'  [{name}] rho(release date, raw value) = {rho:+.2f}, n={len(sub)}')
        for c in cells:
            print('     ', c)
