# Music Artist Recommendation with Last.fm

This project recommends music artists from the HetRec 2011 Last.fm 2K
dataset. It uses implicit feedback: `weight` is a play count and is not a
star rating. The fixed random seed for all later randomized procedures is
`42`.

## Beginner map

The code is split into three small ideas:

1. [`src/data.py`](src/data.py) downloads the files and reads them into
   pandas tables.
2. [`src/evaluation.py`](src/evaluation.py) splits each user's listening
   history, creates recommendations, and calculates the scores.
3. [`src/run_experiments.py`](src/run_experiments.py) connects those steps and
   saves the results.

The recommendation baseline is intentionally simple: count how many training
users listened to each artist, then recommend the most popular artists that
the current user has not already heard. The three evaluation measures mean:

- **Precision**: how many recommendations were relevant.
- **Recall**: how many of the user's held-out artists were found.
- **NDCG**: gives extra credit when relevant artists appear near the top.

## Stage 1: setup and exploration

The Stage 1 command downloads and extracts the dataset into `data/`, loads
the four tab-separated files, builds a sparse user-by-artist interaction
matrix, and saves exploratory outputs under `results/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiments.py
```

The generated outputs are:

- `results/stage1_summary.txt`: counts, sparsity, and distribution statistics.
- `results/cleaning_notes.md`: schema and basic data-quality checks.
- `results/figures/play_count_distribution.png`
- `results/figures/items_per_user_distribution.png`
- `results/environment.txt`: Python, package, hardware, seed, and runtime metadata.

The automatic download uses the GroupLens archive URL. If the network is
unavailable, download `hetrec2011-lastfm-2k.zip` from
<https://grouplens.org/datasets/hetrec-2011/>, extract it under `data/`, and
rerun the command.

## Dataset files

The loader follows the archive README schemas:

- `user_artists.dat`: `userID`, `artistID`, `weight`
- `artists.dat`: `id`, `name`, `url`, `pictureURL`
- `tags.dat`: `tagID`, `tagValue`
- `user_taggedartists.dat`: `userID`, `artistID`, `tagID`, `day`, `month`, `year`

The artist and tag text files are read as Latin-1 because the archive contains
non-UTF-8 characters in some names and tag values.

## Public repository note

The raw dataset is excluded by `.gitignore` and should not be committed to
this repository. The HetRec 2011 README says that the data is available for
non-commercial use, so keep that restriction and the dataset attribution when
using or sharing the project.

## Stage 2: splits, metrics, and popularity baseline

Run Stage 2 with:

```bash
python src/run_experiments.py --stage 2
```

The split is deterministic with seed 42 and is made separately for every
user. Users with fewer than five interactions are excluded. Eligible users
receive approximately 70% train, 10% validation, and 20% test interactions,
with at least one item in each split. The validation and test sets are never
used to build popularity scores.

The candidate set is all artists appearing in training data, minus the
artists already in that user's training history. Training artists are never
recommended. Artists held out from a user but absent from all training data
are reported as unreachable for collaborative filtering.

Stage 2 saves:

- `results/stage2_summary.json`
- `results/popularity_validation_metrics.csv`
- `results/popularity_test_metrics.csv`

The popularity baseline ranks artists by the number of eligible training
users who listened to them. Precision@K, Recall@K, and NDCG@K are reported
for K=5, 10, and 20.

## Later stages

Stage 2 will add per-user train/validation/test splits, a popularity
baseline, metrics, and tests. Stage 3 will add item-based collaborative
filtering and implicit-feedback ALS. The candidate set will be all artists
seen in training, excluding every artist already in that user's training
history; training items must never be recommended. Stages 4–7 will add final
test results, error analysis, optional tag features, and a FastAPI service.
