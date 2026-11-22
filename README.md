# Hallucination leaderboards: data and analysis

Data and scripts behind the Wondering About AI article "Do today's AI models hallucinate less?" by Karen Spinner.

The article compares public hallucination leaderboards and sorts their models by release date. This repository holds the leaderboard tables as collected in November 2026, the analysis scripts and the charts.

## Disclosure

This is a personal project. Karen Spinner has no connection to any of the leaderboards. Claude models ran the research and wrote the scripts. GPT-6.1 Sol reviewed the analysis and the draft four times. Both Anthropic's and OpenAI's models are ranked on these leaderboards.

## What's here

| Path | Contents |
|---|---|
| `data/` | One CSV per leaderboard, long format: `board, metric, higher_is_better, model_raw, model_canonical, vendor, setting, value, snapshot_date, source_url, note` |
| `analysis/identity.py` | Model matching. Separates distinct releases of a model and assigns release dates |
| `analysis/trend_by_release.py` | Scores by release half-year within each leaderboard |
| `analysis/corr.py` | Rank correlations between leaderboards over the models they share |
| `analysis/article_figures.py` | Recomputes every data-derived figure quoted in the article, including the comparison of each model with earlier releases in its product line |
| `analysis/*_output.txt`, `analysis/corr_*.json` | Saved output of those scripts |
| `charts/chart_data.py`, `charts/data.json` | The numbers behind the five charts |
| `charts/template.html`, `charts/build.py`, `charts/charts.html` | The charts as a web page |
| `charts/export_png.py`, `charts/png/` | The five chart images used in the article |

## Leaderboard tables

| File | Leaderboard | Source | Snapshot |
|---|---|---|---|
| `vectara_new.csv` | Vectara Hallucination Leaderboard, current dataset | github.com/vectara/hallucination-leaderboard | 2026-10-22 |
| `vectara_old.csv` | Vectara Hallucination Leaderboard, original dataset | same repository, tag `hhem-2.3-old-dataset` | 2025-10-16 |
| `faithjudge.csv` | FaithJudge | github.com/vectara/FaithJudge | 2026-11-12 |
| `bullshitbench_v2.csv` | BullshitBench v2 | github.com/petergpt/bullshit-benchmark | 2026-11-11 |
| `simpleqa_verified_epoch.csv` | SimpleQA Verified, Epoch AI runs (current and retired prompt) | epoch.ai/benchmarks/simpleqa-verified (CC BY) | 2026-11-12 and 2026-08-22 |
| `simpleqa_verified_kaggle.csv` | SimpleQA Verified, Kaggle board | kaggle.com/benchmarks/deepmind/simpleqa-verified | 2026-11-12 |
| `facts_v2.csv` | FACTS Benchmark Suite | kaggle.com/benchmarks/google/facts | 2026-11-12 |
| `phare.csv` | Phare, hallucination module | phare.giskard.ai | 2026-11-12 |
| `halluhard.csv` | HalluHard | halluhard.com | 2026-11-12 |
| `lechmazur_confabulation.csv` | Confabulations leaderboard | github.com/lechmazur/confabulations | 2026-11-12 |
| `hallulens.csv` | HalluLens paper tables | arxiv.org/abs/2504.17550 | 2025-04-24 |

The numbers belong to the leaderboards' maintainers. Each row keeps its source URL. `model_canonical`, `vendor` and `setting` were added during collection and contain judgment calls.

## AA-Omniscience is not included

The article also uses two tables from Artificial Analysis's AA-Omniscience benchmark (artificialanalysis.ai/evaluations/omniscience): version 1 as of 2026-10-19 and version 1.1 as of 2026-11-12. They are not redistributed here.

To reproduce the AA-Omniscience figures, obtain those tables from Artificial Analysis and save them as `data/aa_omniscience_v1.csv` (board id `aa_v1`) and `data/aa_omniscience_v11.csv` (board id `aa_v11`) in the format above, with the metrics `net_index`, `accuracy`, `hallucination_rate` and `attempt_rate` in percent, and `released=YYYY-MM-DD` in the note column.

Without them, `trend_by_release.py` and `corr.py` run on the other leaderboards, `article_figures.py` skips the AA-Omniscience figures, and `chart_data.py` stops with a message, because three of the five charts need those tables. The saved outputs in `analysis/` were produced with the AA-Omniscience tables present.

## Running the scripts

Python 3 with NumPy.

```
python3 analysis/trend_by_release.py data analysis/bullshitbench_launch_dates.csv
```

```
python3 analysis/corr.py data analysis/corr_mean.json mean
```

```
python3 analysis/article_figures.py data analysis/bullshitbench_launch_dates.csv
```

To rebuild the charts (the image export needs Google Chrome and Pillow):

```
python3 charts/chart_data.py data analysis/bullshitbench_launch_dates.csv charts/data.json
```

```
python3 charts/build.py
```

```
python3 charts/export_png.py
```

## Licence

The scripts and charts are released under the MIT licence (see `LICENSE`).

The leaderboard tables are not covered by that licence. They belong to their maintainers and are reproduced here with attribution, under the maintainers' own terms where stated: Vectara's hallucination leaderboard (Apache-2.0), BullshitBench (MIT) and Epoch AI's benchmark data (CC BY). The other sources state no licence. If you maintain one of them and want a table removed, open an issue.

## Limits

- Each table is a snapshot. Leaderboards change their questions, judges and model lists, so scores from different versions are not comparable.
- Leaderboards choose which models to test. Groups of models by release date are selections, not a history.
- Model names are matched across leaderboards by rule. Some true matches are missed, and a leaderboard that does not say which release it tested can be matched to the wrong one.
- A model's reasoning settings are averaged unless a script says otherwise.
- Product lines are grouped by a name rule in `article_figures.py`. A stricter grouping of OpenAI's models moves the counts by one or two lines.
