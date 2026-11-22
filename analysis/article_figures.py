"""Recompute every data-derived figure quoted in the article.

Usage: python3 -I article_figures.py <data_dir> <bullshitbench_launch_csv>
Models are release-separated identities (see identity.py); a model's settings are averaged.
"""
import json, os, re, statistics as st, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import identity
import numpy as np

data_dir, bsb_launch = sys.argv[1], sys.argv[2]
rows, rel, conflicts = identity.load(data_dir, bsb_launch)
BIG = ('openai', 'anthropic', 'google', 'xai', 'meta')
HALVES = ['2025H1', '2025H2', '2026H1', '2026H2']


def half(d):
    return d[:4] + ('H1' if int(d[5:7]) <= 6 else 'H2')


def is_big(v):
    return any(b in v for b in BIG)


def table(board, metric, drop=lambda r: False):
    vals, vend = defaultdict(list), {}
    for r in rows:
        if r['board'] == board and r['metric'] == metric and not drop(r):
            try:
                vals[r['ident']].append(float(r['value']))
            except ValueError:
                continue
            vend[r['ident']] = r['vendor'].lower()
    return {k: st.mean(v) for k, v in vals.items()}, vend


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
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def by_half(vals, vend, big=True):
    out = defaultdict(list)
    for k, v in vals.items():
        if k in rel and (not big or is_big(vend[k])):
            out[half(rel[k])].append(v)
    return out


def aa_shares(board='aa_v1'):
    """Correct, attempted, wrong and partial shares per model. Computed inside each configuration that reports
    accuracy, net index and attempt rate (wrong = accuracy - net index), then averaged across a model's settings."""
    cfg = defaultdict(dict)
    for r in rows:
        if r['board'] == board and r['metric'] in ('accuracy', 'net_index', 'attempt_rate'):
            try:
                cfg[(r['ident'], r['model_raw'])][r['metric']] = float(r['value'])
            except ValueError:
                continue
    per = defaultdict(list)
    for (ident, _), m in cfg.items():
        if len(m) == 3:
            wrong = m['accuracy'] - m['net_index']
            per[ident].append((m['accuracy'], m['attempt_rate'], wrong, m['attempt_rate'] - m['accuracy'] - wrong))
    return {k: tuple(st.mean(c[i] for c in v) for i in range(4)) for k, v in per.items()}


def days(d):
    return int(d[:4]) * 372 + int(d[5:7]) * 31 + int(d[8:])


def line(label, value):
    print(f'{label:<74} {value}')


# The AA-Omniscience tables are not redistributed (see README). Without them, the sections that need them are skipped.
HAS_AA = any(r['board'] == 'aa_v1' for r in rows)
if not HAS_AA:
    print('NOTE: AA-Omniscience tables not found in the data folder; AA-Omniscience figures are skipped.\n')


def have(*boards):
    return [b for b in boards if HAS_AA or not b.startswith('aa_')]


print('== model counts (release-separated identities)')
for b, m in [('aa_v1', 'accuracy'), ('simpleqa_verified_epoch', 'accuracy'), ('bullshitbench_v2', 'green_rate'),
             ('vectara_new', 'hallucination_rate'), ('facts_v2', 'grounding_score')]:
    line(b, len(table(b, m)[0]))

print('\n== Finding 1: accuracy, big-lab median by release half-year (n)')
for b in have('aa_v1', 'simpleqa_verified_epoch'):
    vals, vend = table(b, 'accuracy')
    h = by_half(vals, vend)
    scale = 100 if b.startswith('simpleqa') else 1
    line(b + ' medians', [(x, round(st.median(h[x]) * scale, 1), len(h[x])) for x in HALVES])
    line(b + ' best 2025H1 / 2026H2', (round(max(h['2025H1']) * scale, 1), round(max(h['2026H2']) * scale, 1)))
    if b == 'simpleqa_verified_epoch':
        line('  vendors in 2025H1', sorted({vend[k] for k in vals if k in rel and is_big(vend[k]) and half(rel[k]) == '2025H1'}))
    for lab in BIG:
        pts = [(days(rel[k]), v) for k, v in vals.items() if k in rel and lab in vend[k]]
        if len(pts) >= 4:
            line(f'  {lab}: rho(release date, accuracy), n', (round(spearman([p[0] for p in pts], [p[1] for p in pts]), 2), len(pts)))

print('\n== Finding 1: range inside each half-year, big labs (min, median, max)')
for b in have('aa_v1', 'simpleqa_verified_epoch'):
    vals, vend = table(b, 'accuracy')
    h = by_half(vals, vend)
    scale = 100 if b.startswith('simpleqa') else 1
    line(b, [(x, round(min(h[x]) * scale, 1), round(st.median(h[x]) * scale, 1), round(max(h[x]) * scale, 1)) for x in HALVES])
    line(b + ' best model per half-year', [(x, max((v, k) for k, v in vals.items() if k in rel and is_big(vend[k]) and half(rel[k]) == x)[1]) for x in HALVES])

print('\n== Finding 2: BullshitBench, big-lab medians 2025H1 vs 2026H2 (n)')
for m in ('red_rate', 'green_rate'):
    vals, vend = table('bullshitbench_v2', m)
    h = by_half(vals, vend)
    line(m, [(x, round(st.median(h[x]) * 100, 1), len(h[x])) for x in ('2025H1', '2026H2')])
g, vend = table('bullshitbench_v2', 'green_rate')
gx, _ = table('bullshitbench_v2', 'green_rate_excl_refusals')
for name, t in (('Claude pushback median', g), ('Claude pushback excl. refusals', gx)):
    h = defaultdict(list)
    for k, v in t.items():
        if 'anthropic' in vend.get(k, '') and k in rel:
            h[half(rel[k])].append(v)
    line(name + ' 2026H1 / 2026H2', (round(st.median(h['2026H1']) * 100, 1), round(st.median(h['2026H2']) * 100, 1), len(h['2026H1']), len(h['2026H2'])))
ref = [float(r['value']) for r in rows if r['board'] == 'bullshitbench_v2' and r['metric'] == 'refusal_rate'
       and 'anthropic' in r['vendor'].lower() and float(r['value']) > 0]
line('Claude configurations with refusals, max refusal', (len(ref), max(ref)))

print('\n== Finding 3: AA-Omniscience v1')
if HAS_AA:
    hr, vend = table('aa_v1', 'hallucination_rate')
    at, _ = table('aa_v1', 'attempt_rate')
    ac, _ = table('aa_v1', 'accuracy')
    h = by_half(hr, vend, big=False)
    line('hallucination rate median, all models', [(x, round(st.median(h[x]), 1), len(h[x])) for x in HALVES])
    h = by_half(at, vend, big=False)
    line('attempt rate median, all models', [(x, round(st.median(h[x]), 1), len(h[x])) for x in ('2026H1', '2026H2')])
    sh = aa_shares()
    for name, big in (('all models', False), ('five big labs', True), ('all other companies', 'other')):
        parts = {}
        for hv in ('2026H1', '2026H2'):
            ks = [k for k in sh if k in rel and half(rel[k]) == hv and (not big or is_big(vend[k]) == (big is True))]
            parts[hv] = tuple(st.mean(sh[k][i] for k in ks) for i in range(4)) + (len(ks),)
        a, b = parts['2026H1'], parts['2026H2']
        line(f'{name}: wrong answers mean', (round(a[2], 2), round(b[2], 2), 'diff', round(a[2] - b[2], 2), 'n', a[4], b[4]))
        line(f'{name}: fewer attempts / more correct / more partial',
             (round(a[1] - b[1], 2), round(b[0] - a[0], 2), round(b[3] - a[3], 2)))
    for raw in ('Claude Fable 5.1', 'Gemini 4 Hydrogen'):
        cfg = defaultdict(dict)
        for r in rows:
            if r['board'] == 'aa_v11' and r['model_raw'].startswith(raw):
                cfg[r['model_raw']][r['metric']] = round(float(r['value']), 2)
        for k, v in cfg.items():
            line('  aa_v11 ' + k, (v.get('net_index'), v.get('attempt_rate')))

print('\n== Finding 4: Vectara new board')
vh, vend = table('vectara_new', 'hallucination_rate')
vl, _ = table('vectara_new', 'avg_summary_length_words')
h = by_half(vh, vend)
line('big-lab median by half-year', [(x, round(st.median(h[x]), 2), len(h[x])) for x in HALVES])
ks = [k for k in vh if k in rel and is_big(vend[k])]
line('rho(release date, rate), big labs, n', (round(spearman([days(rel[k]) for k in ks], [vh[k] for k in ks]), 3), len(ks)))
# rank regression: rate ~ release date, controlling for lab and summary length
y = np.array(rank([vh[k] for k in ks])); d = np.array(rank([days(rel[k]) for k in ks]))
L = np.array(rank([vl[k] for k in ks]))
labs = sorted({next(b for b in BIG if b in vend[k]) for k in ks})
D = np.array([[1.0 if lab in vend[k] else 0.0 for lab in labs] for k in ks])


def resid(v, X):
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


X = np.column_stack([D, L])
line('  partial rho controlling for lab and summary length', round(float(np.corrcoef(resid(y, X), resid(d, X))[0, 1]), 3))
ok = [k for k in vh if 'openai' in vend[k]]
okd = [k for k in ok if k in rel]
line('OpenAI models on board / with release date', (len(ok), len(okd)))
line('OpenAI rho(release date, rate)', round(spearman([days(rel[k]) for k in okd], [vh[k] for k in okd]), 3))
oh = defaultdict(list)
for k in okd:
    oh[half(rel[k])].append(vh[k])
line('OpenAI median by half-year', [(x, round(st.median(oh[x]), 1), len(oh[x])) for x in HALVES if oh[x]])
for lab in BIG:
    lk = [k for k in vh if lab in vend[k] and k in rel]
    lhh = defaultdict(list)
    for k in lk:
        lhh[half(rel[k])].append(vh[k])
    rho = round(spearman([days(rel[k]) for k in lk], [vh[k] for k in lk]), 2) if len(lk) >= 4 else None
    line(f'  {lab}: n, rho(release date, rate), median by half-year (n)',
         (len(lk), rho, [(x, round(st.median(lhh[x]), 1), len(lhh[x])) for x in HALVES if lhh[x]]))
line('labs present per half-year', [(x, sorted({next(b for b in BIG if b in vend[k]) for k in vh
                                                 if k in rel and is_big(vend[k]) and half(rel[k]) == x})) for x in HALVES])
lh = by_half(vl, vend)
line('big-lab summary length median 2025H1 / 2026H2', (round(st.median(lh['2025H1']), 1), round(st.median(lh['2026H2']), 1)))
allk = [k for k in vh if k in vl]
line('rho(summary length, rate), all models, n', (round(spearman([vl[k] for k in allk], [vh[k] for k in allk]), 3), len(allk)))
fg, fv = table('facts_v2', 'grounding_score')
fk = [k for k in fg if k in rel]
line('FACTS grounding: n, max, rho(release date, score), n released 2026',
     (len(fg), round(max(fg.values()), 1), round(spearman([days(rel[k]) for k in fk], [fg[k] for k in fk]), 3),
      sum(1 for k in fk if rel[k].startswith('2026'))))

print('\n== Finding 5: rank correlations (from corr_mean.json)')
pairs = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corr_mean.json')))['pairs']
P = {frozenset((p['a'], p['b'])): p for p in pairs}
fact = ['AA v1: accuracy', 'AA v1.1: accuracy', 'SimpleQA-V Kaggle: score', 'SimpleQA-V Epoch: forced-guess acc',
        'FACTS: parametric', 'Phare: factuality']
rhos = []
for i, a in enumerate(fact):
    for b in fact[i + 1:]:
        p = P.get(frozenset((a, b)))
        if p and p['rho'] is not None and not p['same_board'] and p['n'] >= 15 and not (a.startswith('AA') and b.startswith('AA')):
            rhos.append((p['rho'], p['n'], a, b))
line('fact-recall tests, min and max rho (n>=15)', (min(rhos), max(rhos)))
for a, b in (('Vectara new: halluc rate', 'AA v1: halluc rate'), ('Vectara new: halluc rate', 'FACTS: grounding'),
             ('Vectara new: halluc rate', 'Vectara old: halluc rate')):
    p = P[frozenset((a, b))]
    line(f'{a} x {b}', (p['rho'], p.get('lo'), p.get('hi'), 'n', p['n']))

print('\n== Each model against its predecessor in the same product line (big labs)')
SKIP = re.compile(r'gemma|llama|gpt-oss')  # names mix model sizes, so a later release is not a successor


def family(name):
    """Product line: the model name without version numbers, e.g. claude-opus-4.5 -> claude-opus."""
    return '-'.join(t for t in re.split(r'[-\s]+', name.lower()) if not re.fullmatch(r'[\d.]+[a-z]?|\d+k|preview|exp|latest', t))


for b, m, hib in (('aa_v1', 'accuracy', True), ('simpleqa_verified_epoch', 'accuracy', True), ('bullshitbench_v2', 'green_rate', True),
                  ('bullshitbench_v2', 'red_rate', False), ('vectara_new', 'hallucination_rate', False)):
    if b not in have(b):
        continue
    vals, vend = table(b, m)
    name = {r['ident']: r['model_canonical'] for r in rows if r['board'] == b}
    lines = defaultdict(list)
    for k, v in vals.items():
        if k in rel and is_big(vend[k]) and not SKIP.match(family(name[k])):
            lines[(vend[k], family(name[k]))].append((rel[k], v, name[k]))
    sign = 1 if hib else -1
    lines = {f: sorted(ms) for f, ms in lines.items() if len(ms) > 1}
    steps = [(y[1] - x[1]) * sign for ms in lines.values() for x, y in zip(ms, ms[1:]) if x[0] != y[0]]
    ends = {f: (ms[-1][1] - ms[0][1]) * sign for f, ms in lines.items()}
    line(f'{b} {m}: lines, newest beats oldest, steps improved / steps',
         (len(lines), sum(e > 0 for e in ends.values()), sum(p > 0 for p in steps), len(steps)))
    line('  lines where the newest does not beat the oldest', sorted(f[1] for f, e in ends.items() if e <= 0))

print('\n== Limitations')
cross = [c for c in conflicts if len({half(d) for d in c[1]}) > 1]
line('identities with conflicting release dates / crossing a half-year', (len(conflicts), len(cross)))
for b, m, sign in (('vectara_new', 'hallucination_rate', -1), ('bullshitbench_v2', 'green_rate', 1)):
    vals, vend = table(b, m)
    order = sorted(vals, key=lambda k: sign * vals[k])
    pct = [100 * order.index(k) / (len(order) - 1) for k in vals if 'anthropic' in vend[k]]
    line(f'Anthropic median percentile on {b} (100 = best)', round(st.median(pct), 0))
for b, m, scale in (('aa_v1', 'accuracy', 1), ('bullshitbench_v2', 'green_rate', 100)):
    if b not in have(b):
        continue
    vals, vend = table(b, m)
    h = defaultdict(list)
    for k, v in vals.items():
        if k in rel and is_big(vend[k]) and 'anthropic' not in vend[k]:
            h[half(rel[k])].append(v)
    line(f'{b} {m} big labs excluding Anthropic, 2025H1 / 2026H2', (round(st.median(h['2025H1']) * scale, 1), round(st.median(h['2026H2']) * scale, 1)))
