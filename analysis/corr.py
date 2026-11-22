"""Cross-leaderboard Spearman correlations for hallucination benchmarks.

Usage: python3 -I corr.py <data_dir> <out_json> [collapse=mean|best]
Every metric is oriented so that higher = better before ranking.
"""
import csv, glob, json, os, re, sys, random
from collections import defaultdict

data_dir, out_json = sys.argv[1], sys.argv[2]
collapse = sys.argv[3] if len(sys.argv) > 3 else 'mean'

# (board, metric) -> label, sign (+1 higher is better, -1 lower is better), construct
SERIES = {
    ('vectara_new', 'hallucination_rate'): ('Vectara new: halluc rate', -1, 'grounded'),
    ('vectara_old', 'hallucination_rate'): ('Vectara old: halluc rate', -1, 'grounded'),
    ('faithjudge', 'hallucination_rate'): ('FaithJudge: halluc rate', -1, 'grounded'),
    ('lechmazur', 'confabulation_rate'): ('lechmazur: confab rate', -1, 'grounded'),
    ('lechmazur', 'weighted_score'): ('lechmazur: weighted', -1, 'grounded'),
    ('facts_v2', 'grounding_score'): ('FACTS: grounding', 1, 'grounded'),
    ('halluhard', 'hallucination_rate'): ('HalluHard: halluc rate', -1, 'grounded'),
    ('facts_v2', 'parametric_score'): ('FACTS: parametric', 1, 'closed-book'),
    ('facts_v2', 'search_score'): ('FACTS: search', 1, 'search'),
    ('facts_v2', 'facts_score'): ('FACTS: overall', 1, 'mixed'),
    ('aa_v11', 'net_index'): ('AA v1.1: net index', 1, 'closed-book'),
    ('aa_v11', 'accuracy'): ('AA v1.1: accuracy', 1, 'closed-book'),
    ('aa_v11', 'hallucination_rate'): ('AA v1.1: halluc rate', -1, 'closed-book'),
    ('aa_v1', 'net_index'): ('AA v1: net index', 1, 'closed-book'),
    ('aa_v1', 'accuracy'): ('AA v1: accuracy', 1, 'closed-book'),
    ('aa_v1', 'hallucination_rate'): ('AA v1: halluc rate', -1, 'closed-book'),
    ('simpleqa_verified_kaggle', 'score'): ('SimpleQA-V Kaggle: score', 1, 'closed-book'),
    ('simpleqa_verified_epoch', 'accuracy'): ('SimpleQA-V Epoch: forced-guess acc', 1, 'closed-book'),
    ('simpleqa_verified_epoch_pre', 'accuracy'): ('SimpleQA-V Epoch pre: acc', 1, 'closed-book'),
    ('bullshitbench_v2', 'green_rate'): ('BullshitBench: green rate', 1, 'false-premise'),
    ('bullshitbench_v2', 'red_rate'): ('BullshitBench: red rate', -1, 'false-premise'),
    ('phare', 'hallucination_resistance'): ('Phare: resistance', 1, 'mixed'),
    ('phare', 'factuality'): ('Phare: factuality', 1, 'closed-book'),
    ('phare', 'misinformation'): ('Phare: misinformation', 1, 'false-premise'),
    ('phare', 'debunking'): ('Phare: debunking', 1, 'false-premise'),
    # descriptive abstention columns, oriented as "answers more"
    ('aa_v11', 'attempt_rate'): ('AA v1.1: attempt rate', 1, 'abstention'),
    ('aa_v1', 'attempt_rate'): ('AA v1: attempt rate', 1, 'abstention'),
    ('vectara_new', 'answer_rate'): ('Vectara new: answer rate', 1, 'abstention'),
    ('vectara_new', 'avg_summary_length_words'): ('Vectara new: summary length', 1, 'abstention'),
    ('lechmazur', 'non_response_rate'): ('lechmazur: non-response', -1, 'abstention'),
}


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import identity  # shared model identity: canonical name plus release tag

all_rows, _rel, _conflicts = identity.load(data_dir)
rows = defaultdict(lambda: defaultdict(list))  # series -> model identity -> [values]
names = defaultdict(dict)
for r in all_rows:
    s = (r['board'], r['metric'])
    if s not in SERIES:
        continue
    setting = (r.get('setting') or '').lower()
    if r['board'] == 'halluhard' and 'websearch' in setting:
        continue  # different condition
    try:
        v = float(r['value'])
    except (ValueError, TypeError):
        continue
    k = r['ident']
    rows[s][k].append(v * SERIES[s][1])
    names[r['board']][k] = r['model_canonical']

series = {}
for s, d in rows.items():
    if collapse == 'best':
        series[s] = {k: max(v) for k, v in d.items()}
    else:
        series[s] = {k: sum(v) / len(v) for k, v in d.items()}


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = sum((x - ma) ** 2 for x in a) ** 0.5
    sb = sum((y - mb) ** 2 for y in b) ** 0.5
    if sa == 0 or sb == 0:
        return float('nan')
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def spearman(a, b):
    return pearson(rank(a), rank(b))


def boot_ci(a, b, n_boot=2000, seed=7):
    rng = random.Random(seed)
    n = len(a)
    out = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        r = spearman([a[i] for i in idx], [b[i] for i in idx])
        if r == r:
            out.append(r)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]


def partial(a, b, c):
    """Spearman partial correlation of a,b controlling for c."""
    ra, rb, rc = rank(a), rank(b), rank(c)
    ab, ac, bc = pearson(ra, rb), pearson(ra, rc), pearson(rb, rc)
    den = ((1 - ac ** 2) * (1 - bc ** 2)) ** 0.5
    return (ab - ac * bc) / den if den else float('nan')


pairs = []
keys = sorted(series)
for i, s1 in enumerate(keys):
    for s2 in keys[i + 1:]:
        if s1[0] == s2[0]:
            same_board = True
        else:
            same_board = False
        shared = sorted(set(series[s1]) & set(series[s2]))
        if len(shared) < 8:
            pairs.append({'a': SERIES[s1][0], 'b': SERIES[s2][0], 'n': len(shared), 'rho': None,
                          'same_board': same_board, 'ca': SERIES[s1][2], 'cb': SERIES[s2][2]})
            continue
        a = [series[s1][k] for k in shared]
        b = [series[s2][k] for k in shared]
        rho = spearman(a, b)
        lo, hi = boot_ci(a, b)
        pairs.append({'a': SERIES[s1][0], 'b': SERIES[s2][0], 'n': len(shared), 'rho': round(rho, 3),
                      'lo': round(lo, 2), 'hi': round(hi, 2), 'same_board': same_board,
                      'ca': SERIES[s1][2], 'cb': SERIES[s2][2]})

# capability control: AA v1 accuracy as the widest-coverage knowledge proxy
ctrl = series.get(('aa_v1', 'accuracy'), {})
partials = []
for i, s1 in enumerate(keys):
    for s2 in keys[i + 1:]:
        if s1[0] == s2[0] or 'aa_v1' in (s1[0], s2[0]):
            continue
        shared = sorted(set(series[s1]) & set(series[s2]) & set(ctrl))
        if len(shared) < 12:
            continue
        a = [series[s1][k] for k in shared]
        b = [series[s2][k] for k in shared]
        c = [ctrl[k] for k in shared]
        partials.append({'a': SERIES[s1][0], 'b': SERIES[s2][0], 'n': len(shared),
                         'rho': round(spearman(a, b), 3), 'partial_given_aa_v1_accuracy': round(partial(a, b, c), 3)})

json.dump({'collapse': collapse, 'pairs': pairs, 'partials': partials,
           'models_per_series': {SERIES[s][0]: len(series[s]) for s in keys},
           'names': {b: sorted(set(v.values())) for b, v in names.items()}},
          open(out_json, 'w'), indent=1)

print('collapse =', collapse)
print('\nMODELS PER SERIES')
for s in keys:
    print(f'  {len(series[s]):4d}  {SERIES[s][0]}')
print('\nCROSS-BOARD PAIRS (n>=8), sorted by rho')
cross = [p for p in pairs if not p['same_board'] and p['rho'] is not None]
for p in sorted(cross, key=lambda p: -p['rho']):
    print(f"  {p['rho']:+.2f} [{p['lo']:+.2f},{p['hi']:+.2f}] n={p['n']:3d}  {p['a']}  x  {p['b']}  ({p['ca']}/{p['cb']})")
print('\nSAME-BOARD PAIRS')
for p in sorted([p for p in pairs if p['same_board'] and p['rho'] is not None], key=lambda p: -p['rho']):
    print(f"  {p['rho']:+.2f} [{p['lo']:+.2f},{p['hi']:+.2f}] n={p['n']:3d}  {p['a']}  x  {p['b']}")
print('\nPARTIALS controlling for AA v1 accuracy (n>=12)')
for p in sorted(partials, key=lambda p: -abs(p['rho'] - p['partial_given_aa_v1_accuracy'])):
    print(f"  rho {p['rho']:+.2f} -> partial {p['partial_given_aa_v1_accuracy']:+.2f} n={p['n']:3d}  {p['a']}  x  {p['b']}")
print('\nTOO FEW SHARED MODELS (<8):', sum(1 for p in pairs if p['rho'] is None and not p['same_board']))
