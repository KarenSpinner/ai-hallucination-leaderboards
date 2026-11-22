"""Shared model identity and release-date logic for the leaderboard tables.

A model identity is the canonical name plus a release tag when a board lists more than one
release of that name (for example deepseek-r1 and deepseek-r1 0528). Reasoning settings,
thinking budgets and hosting labels are not releases; rows that differ only in those are
averaged by the callers.
"""
import csv, glob, os, re
from collections import defaultdict

MONTHS = {m: i + 1 for i, m in enumerate(
    ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}
MONTHS.update({'june': 6, 'july': 7, 'sept': 9})
# names where "thinking" is a separately released model, not a setting
THINKING_IS_RELEASE = {key_name for key_name in ('2 k kimi',)}


def key(name):
    """Order-insensitive token key: claude-4.5-sonnet == claude-sonnet-4.5; '+' is its own token."""
    n = name.lower().replace('+', ' plus ')
    toks = re.findall(r'[a-z]+|\d+(?:\.\d+)?', n)
    toks = [re.sub(r'\.0$', '', t) for t in toks if t not in ('nvidia', 'tencent')]
    return ' '.join(sorted(toks))


def release_tag(setting, canon_key):
    """Reduce a free-text setting to a release tag: month of a date stamp, exp, or a build number."""
    s = setting.lower()
    tags = set()
    for m in re.finditer(r"(jan|feb|mar|apr|may|june|jun|july|jul|aug|sept|sep|oct|nov|dec)[a-z]*[\s'-]*(\d{2,4})?", s):
        tags.add(f'm{MONTHS[m.group(1)]:02d}')
    s2 = re.sub(r'\d+k\b', ' ', s)  # thinking budgets such as 16k, 32k
    for m in re.finditer(r'(?<![\d.])(\d{4})-(\d{2})-(\d{2})(?!\d)', s2):
        tags.add(f'm{m.group(2)}')
    s2 = re.sub(r'(?<![\d.])\d{4}-\d{2}-\d{2}(?!\d)', ' ', s2)
    for m in re.finditer(r'(?<![\d.])(20\d{2})(\d{2})(\d{2})(?!\d)', s2):   # 20250528
        tags.add(f'm{m.group(2)}')
    s2 = re.sub(r'(?<![\d.])20\d{6}(?!\d)', ' ', s2)
    for m in re.finditer(r'(?<![\d.])(2[4-7])(\d{2})(\d{2})(?!\d)', s2):   # 260215
        tags.add(f'm{m.group(2)}')
    s2 = re.sub(r'(?<![\d.])2[4-7]\d{4}(?!\d)', ' ', s2)
    for m in re.finditer(r'(?<![\d.])(20\d{2})-(\d{2})(?!\d)', s2):        # 2025-02
        tags.add(f'm{m.group(2)}')
    s2 = re.sub(r'(?<![\d.])20\d{2}-\d{2}(?!\d)', ' ', s2)
    for m in re.finditer(r'(?<![\d.])(\d{2})-(20\d{2})(?!\d)', s2):        # 03-2025
        tags.add(f'm{m.group(1)}')
    s2 = re.sub(r'(?<![\d.])\d{2}-20\d{2}(?!\d)', ' ', s2)
    for m in re.finditer(r'(?<![\d.])(\d{2})-(\d{2})(?!\d)', s2):          # 06-05
        if 1 <= int(m.group(1)) <= 12:
            tags.add(f'm{m.group(1)}')
    s2 = re.sub(r'(?<![\d.])\d{2}-\d{2}(?!\d)', ' ', s2)
    for m in re.finditer(r'(?<![\d.])(\d{4})(?![\d.])', s2):               # 0528 (MMDD) or 2512 (YYMM)
        a, b = m.group(1)[:2], m.group(1)[2:]
        if a in ('24', '25', '26', '27') and 1 <= int(b) <= 12:
            tags.add(f'm{b}')
        elif 1 <= int(a) <= 12:
            tags.add(f'm{a}')
    for m in re.finditer(r'(?<![\d.])(00\d)(?!\d)', s2):
        tags.add('b' + m.group(1))
    for m in re.finditer(r'v(\d\.\d)', s2):
        tags.add('v' + m.group(1))
    if re.search(r'\bexp\b|exp-', s2):
        tags.add('exp')
    if canon_key in THINKING_IS_RELEASE and 'thinking' in s:
        tags.add('thinking')
    return '+'.join(sorted(tags))


def load(data_dir, bsb_launch=None):
    """Return (rows, release_dates). Each row gains 'ident'. release_dates maps ident -> YYYY-MM-DD."""
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, '*.csv'))):
        with open(path, newline='') as f:
            rows += list(csv.DictReader(f))
    tags_per = defaultdict(set)
    for r in rows:
        r['_ck'] = key(r['model_canonical'])
        r['_tag'] = release_tag(r.get('setting') or '', r['_ck'])
        tags_per[(r['board'], r['_ck'])].add(r['_tag'])
    for r in rows:
        multi = len(tags_per[(r['board'], r['_ck'])]) > 1
        r['ident'] = r['_ck'] + ('|' + r['_tag'] if multi and r['_tag'] else '')

    # release dates, per identity; a canonical-level date is used only when it is unambiguous
    ident_dates, canon_dates = defaultdict(set), defaultdict(set)

    def put(r, d):
        if re.match(r'\d{4}-\d{2}-\d{2}$', d or ''):
            ident_dates[r['ident']].add(d)
            canon_dates[r['_ck']].add(d)
    for r in rows:
        m = re.search(r'released[= ](\d{4}-\d{2}-\d{2})', r['note'])
        if m:
            put(r, m.group(1))
    if bsb_launch:
        by_raw = {}
        for r in rows:
            if r['board'] == 'bullshitbench_v2':
                by_raw[r['model_raw']] = r
        with open(bsb_launch, newline='') as f:
            for x in csv.DictReader(f):
                if x['model'] in by_raw:
                    put(by_raw[x['model']], x['launch_date'])
    rel, conflicts = {}, []
    idents = {r['ident']: r['_ck'] for r in rows}
    for ident, ck in idents.items():
        ds = ident_dates.get(ident)
        if ds:
            if len(ds) > 1:
                conflicts.append((ident, sorted(ds)))
            rel[ident] = min(ds)
        elif len(canon_dates.get(ck, ())) == 1 and '|' not in ident:
            rel[ident] = next(iter(canon_dates[ck]))
    return rows, rel, conflicts
