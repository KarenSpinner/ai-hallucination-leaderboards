"""Compute the numbers behind the article's five charts and write charts/data.json.

Usage: python3 -I charts/chart_data.py <data_dir> <bullshitbench_launch_csv> <out_json>
"""
import json, os, statistics as st, sys
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis'))
import identity

data_dir, bsb_launch, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows, rel, _ = identity.load(data_dir, bsb_launch)
if not any(r['board'] == 'aa_v1' for r in rows):
    sys.exit('The charts need the AA-Omniscience tables, which are not redistributed. See the README.')
BIG = {'openai': 'OpenAI', 'anthropic': 'Anthropic', 'google': 'Google', 'xai': 'xAI', 'meta': 'Meta'}
HALVES = ['2025H1', '2025H2', '2026H1', '2026H2']


def r1(v):
    """Round half up to one decimal, matching the figures quoted in the article."""
    return float(Decimal(repr(round(v, 6))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def half(d):
    return d[:4] + ('H1' if int(d[5:7]) <= 6 else 'H2')


def lab(v):
    return next((n for k, n in BIG.items() if k in v), None)


def table(board, metric):
    vals, vend, name = defaultdict(list), {}, {}
    for r in rows:
        if r['board'] == board and r['metric'] == metric:
            try:
                vals[r['ident']].append(float(r['value']))
            except ValueError:
                continue
            vend[r['ident']] = r['vendor'].lower()
            name[r['ident']] = r['model_canonical']
    return {k: st.mean(v) for k, v in vals.items()}, vend, name


def medians(board, metric, scale=1.0, best=False):
    vals, vend, _ = table(board, metric)
    by = defaultdict(list)
    for k, v in vals.items():
        if k in rel and lab(vend[k]):
            by[half(rel[k])].append(v * scale)
    return [{'half': h, 'value': r1(st.median(by[h])), 'n': len(by[h]), **({'best': r1(max(by[h]))} if best else {})} for h in HALVES]


def top_setting(board, metric, model):
    """Score at the highest reasoning setting the leaderboard lists for this model, or None if it was not tested."""
    found = {r['setting'].split(';')[0].strip(): float(r['value']) for r in rows
             if r['board'] == board and r['metric'] == metric and r['model_canonical'] == model}
    for s in ('max', 'xhigh', 'high', ''):
        if s in found:
            return found[s], s
    return None, None


d = {'halves': HALVES}
d['accuracy'] = {'AA-Omniscience': medians('aa_v1', 'accuracy', best=True),
                 'SimpleQA Verified': medians('simpleqa_verified_epoch', 'accuracy', 100, best=True)}
d['nonsense'] = {'Accepted the nonsense': medians('bullshitbench_v2', 'red_rate', 100),
                 'Clear pushback': medians('bullshitbench_v2', 'green_rate', 100)}

# Shares are computed inside each configuration that reports accuracy, net index and attempt rate
# (wrong = accuracy - net index), then averaged across a model's settings.
_, vend, _ = table('aa_v1', 'accuracy')
cfg = defaultdict(dict)
for r in rows:
    if r['board'] == 'aa_v1' and r['metric'] in ('accuracy', 'net_index', 'attempt_rate'):
        try:
            cfg[(r['ident'], r['model_raw'])][r['metric']] = float(r['value'])
        except ValueError:
            continue
per = defaultdict(list)
for (ident, _), m in cfg.items():
    if len(m) == 3:
        wrong = m['accuracy'] - m['net_index']
        per[ident].append((m['accuracy'], m['attempt_rate'], wrong, m['attempt_rate'] - m['accuracy'] - wrong))
sh = {k: tuple(st.mean(c[i] for c in v) for i in range(4)) for k, v in per.items()}
split = []
for label, big in (('Models from all other companies, taken together', 'other'), ('Models from the five big labs', True)):
    p = {}
    for h in ('2026H1', '2026H2'):
        ks = [k for k in sh if k in rel and half(rel[k]) == h and bool(lab(vend[k])) == (big is True)]
        p[h] = tuple(st.mean(sh[k][i] for k in ks) for i in range(4)) + (len(ks),)
    a, b = p['2026H1'], p['2026H2']
    split.append({'group': label, 'wrong_early': round(a[2], 1), 'wrong_late': round(b[2], 1), 'diff': r1(a[2] - b[2]),
                  'fewer_attempts': round(a[1] - b[1], 1), 'more_correct': round(b[0] - a[0], 1),
                  'more_partial': round(b[3] - a[3], 1), 'n_early': a[4], 'n_late': b[4]})
d['split'] = split

# One recent model per lab. Wrong answers per 100 questions = accuracy - net index.
cards = []
for label, model in (('Claude Fable 5.2', 'claude-fable-5.2'), ('Gemini 4 Carbon', 'gemini-4-carbon'), ('GPT-6.1 Sol', 'gpt-6.1-sol'),
                     ('Grok 4.7', 'grok-4.7'), ('Muse Wonder', 'muse-wonder')):
    acc, s = top_setting('aa_v11', 'accuracy', model)
    net, _ = top_setting('aa_v11', 'net_index', model)
    sq, _ = top_setting('simpleqa_verified_epoch', 'accuracy', model)
    green, _ = top_setting('bullshitbench_v2', 'green_rate', model)
    red, _ = top_setting('bullshitbench_v2', 'red_rate', model)
    assert top_setting('vectara_new', 'hallucination_rate', model)[0] is None
    cards.append({'model': label, 'aa_setting': s, 'aa_right': r1(acc), 'aa_wrong': r1(acc - net),
                  'simpleqa_right': None if sq is None else r1(sq * 100), 'pushback': r1(green * 100), 'accepted': r1(red * 100)})
d['flagships'] = cards
d['vectara_siblings'] = {m: r1(top_setting('vectara_new', 'hallucination_rate', k)[0])
                         for m, k in (('Claude Opus 5.5', 'claude-opus-5.5'), ('Gemini 4 Argon', 'gemini-4-argon'), ('GPT-6 Sol', 'gpt-6-sol'))}

vh, vvend, vname = table('vectara_new', 'hallucination_rate')
dots = [{'model': vname[k], 'lab': lab(vvend[k]), 'date': rel[k], 'value': r1(v)}
        for k, v in vh.items() if k in rel and lab(vvend[k])]
d['vectara'] = {'dots': sorted(dots, key=lambda x: x['date']), 'medians': medians('vectara_new', 'hallucination_rate')}
json.dump(d, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in d.items() if k != 'vectara'}))
print(len(dots), 'vectara dots', dots[0], dots[-1], d['vectara']['medians'])
print(sorted([(x['value'], x['model'], x['date']) for x in dots])[-3:], [x for x in dots if x['model'] in ('gpt-4.1', 'gpt-6-sol')])
