# Last.fm Music Artist Recommender

This is a small music recommendation project using the HetRec 2011 Last.fm
2K dataset. I used it to learn how a recommender system works from beginning
to end: first understanding the data, then making a simple baseline, and
finally comparing it with a more personalized model.

The project uses listening history rather than star ratings. In
`user_artists.dat`, the `weight` column is the number of times a user
listened to an artist. For the recommendation models, I treat listening as
binary feedback: listened means `1`, and not listened means `0`.

## What the project does

There are three stages:

1. **Stage 1** explores the dataset and checks its basic quality.
2. **Stage 2** creates a popularity-based recommendation baseline.
3. **Stage 3** uses item-based collaborative filtering and compares it with
   the popularity baseline.

The important rule throughout the project is to keep the training,
validation, and test data separate. The models are built using training data
only. Validation data is used to compare models, not to build them.

## How the files fit together

- [`src/data.py`](src/data.py) downloads, loads, and summarizes the dataset.
- [`src/evaluation.py`](src/evaluation.py) creates the data splits and
  calculates Precision, Recall, and NDCG.
- [`src/models.py`](src/models.py) contains the item-based collaborative
  filtering model.
- [`src/run_experiments.py`](src/run_experiments.py) runs the different
  stages and saves their results.
- `tests/` contains small tests for the data loading, evaluation, and model
  rules.

## Setup

Create a virtual environment and install the packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the dataset from GroupLens. If the download does not
work, download `hetrec2011-lastfm-2k.zip` from
<https://grouplens.org/datasets/hetrec-2011/> and extract it into `data/`.

## Running the stages

Run Stage 1:

```bash
python src/run_experiments.py --stage 1
```

Run Stage 2:

```bash
python src/run_experiments.py --stage 2
```

Run Stage 3:

```bash
python src/run_experiments.py --stage 3
```

## Stage 1: understanding the data

Stage 1 loads the tab-separated files, builds a sparse user-by-artist
interaction matrix, and creates a few summary files and plots.

The main outputs are:

- `results/stage1_summary.txt`
- `results/cleaning_notes.md`
- `results/figures/play_count_distribution.png`
- `results/figures/items_per_user_distribution.png`
- `results/environment.txt`

The dataset files use these columns:

- `user_artists.dat`: `userID`, `artistID`, `weight`
- `artists.dat`: `id`, `name`, `url`, `pictureURL`
- `tags.dat`: `tagID`, `tagValue`
- `user_taggedartists.dat`: `userID`, `artistID`, `tagID`, `day`, `month`,
  `year`

## Stage 2: popularity baseline

For Stage 2, each user with at least five artists is split into approximately:

- 70% training data
- 10% validation data
- 20% test data

The split uses random seed `42`, so it is repeatable.

The popularity model counts how many training users listened to each artist.
It then recommends the most popular artists that the current user has not
already listened to in their training history.

Stage 2 saves:

- `results/stage2_summary.json`
- `results/popularity_validation_metrics.csv`
- `results/popularity_test_metrics.csv`

## Stage 3: item-based collaborative filtering

Stage 3 builds an artist-to-user matrix from the training split. The matrix
contains `1` when a user listened to an artist and `0` otherwise.

It then calculates cosine similarity between artists. Artists with similar
listener patterns receive higher similarity scores. For each user, a
candidate artist is scored by adding its similarity to every artist in that
user's training history:

```text
candidate score =
    similarity(candidate, artist 1)
  + similarity(candidate, artist 2)
  + ...
```

Artists already in the user's training history are removed from the
recommendations. The model returns up to 20 artists per user and is evaluated
at K values of 5, 10, and 20.

Stage 3 saves:

- `results/item_cf_validation_metrics.csv`
- `results/stage3_comparison.csv`
- `results/stage3_summary.json`

The comparison file contains both models with these columns:

```text
Model, K, Precision, Recall, NDCG
```

## Understanding the metrics

- **Precision** measures how many recommended artists were relevant.
- **Recall** measures how many of the user's hidden relevant artists were
  found.
- **NDCG** gives more credit when relevant artists appear near the beginning
  of the recommendation list.

As K becomes larger, Recall will often increase because the model has more
recommendation slots. Precision can decrease because more artists are added
to the list.

## Results

In my Stage 3 validation run, item-based collaborative filtering performed
better than the popularity baseline at K=5, K=10, and K=20. This is
reasonable because the popularity model gives broadly popular artists,
whereas item-based collaborative filtering uses each user's own listening
history.

The validation results should still be interpreted carefully. Some artists
in the validation set do not appear in any user's training data, so a
training-based recommender cannot recommend them. This is one limitation of
using a train/validation split for collaborative filtering.

## Tests

Run the tests with:

```bash
PYTHONPATH=. ./.venv/bin/pytest tests/
```

The tests check the split behavior, metric calculations, binary feedback,
recommendation filtering, and safe dataset extraction.

## Dataset note

The raw dataset is ignored by `.gitignore` and is not included in this
repository. The HetRec 2011 README says that the data is available for
non-commercial use. Anyone using the dataset should keep that restriction
and the original attribution in mind.

## My learning notes

This project helped me understand that a recommender system is not only a
model. The data split and evaluation rules are just as important. In
particular, using validation or test information while building the model
would make the results look better than they really are. Keeping those
datasets separate is necessary for a fair comparison.
