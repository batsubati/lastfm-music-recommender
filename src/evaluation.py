"""Simple Stage 2 building blocks for the recommendation experiment.

The project uses implicit feedback: an artist is either in a user's history
or it is not. The play count is used only when calculating popularity scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

SEED = 42
K_VALUES = (5, 10, 20)


@dataclass(frozen=True)
class UserSplits:
    """Artist IDs for each user's train, validation, and test sets."""

    train: Dict[int, frozenset[int]]
    validation: Dict[int, frozenset[int]]
    test: Dict[int, frozenset[int]]
    excluded_users: frozenset[int]


def split_per_user(
    user_artists: pd.DataFrame,
    min_interactions: int = 5,
    seed: int = SEED,
) -> UserSplits:
    """Make deterministic approximately 70/10/20 splits for eligible users.

    Users below ``min_interactions`` are excluded entirely. For every eligible
    user, at least one interaction is retained in validation and test, and
    training receives the remainder.
    """
    rng = np.random.default_rng(seed)
    grouped_users = user_artists.groupby("userID", sort=True)["artistID"]
    train: Dict[int, frozenset[int]] = {}
    validation: Dict[int, frozenset[int]] = {}
    test: Dict[int, frozenset[int]] = {}
    excluded: set[int] = set()

    for user_id, artist_values in grouped_users:
        artists = np.asarray(artist_values, dtype=np.int64)
        if len(artists) < min_interactions:
            excluded.add(int(user_id))
            continue

        shuffled = rng.permutation(artists)
        n_test = max(1, int(round(len(shuffled) * 0.20)))
        n_validation = max(1, int(round(len(shuffled) * 0.10)))

        # Keep at least one artist for training too.
        if n_test + n_validation >= len(shuffled):
            n_validation = 1
            n_test = 1

        train_end = len(shuffled) - n_validation - n_test
        user_id = int(user_id)
        train[user_id] = frozenset(shuffled[:train_end])
        validation[user_id] = frozenset(
            shuffled[train_end:train_end + n_validation]
        )
        test[user_id] = frozenset(shuffled[train_end + n_validation:])

    return UserSplits(train, validation, test, frozenset(excluded))


def candidate_artists(
    user_id: int,
    train: Mapping[int, Iterable[int]],
    training_artists: Iterable[int],
) -> list[int]:
    """Return eligible candidates, never including a user's train artists."""
    # Candidate set is restricted to training artists and explicitly excludes
    # every artist already seen in this user's training history.
    seen = set(train.get(user_id, ()))
    return sorted(set(training_artists) - seen)


def precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Return the fraction of the top-k list that is relevant."""
    recommended_set = set(recommended[:k])
    relevant_set = set(relevant)
    hits = len(recommended_set & relevant_set)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Return the fraction of relevant artists retrieved in the top-k list."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    recommended_set = set(recommended[:k])
    hits = len(recommended_set & relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Return binary-relevance NDCG for a top-k recommendation list."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = 0.0
    for rank, artist_id in enumerate(recommended[:k]):
        if artist_id in relevant_set:
            dcg += 1.0 / np.log2(rank + 2)

    ideal_hits = min(k, len(relevant_set))
    ideal_dcg = 0.0
    for rank in range(ideal_hits):
        ideal_dcg += 1.0 / np.log2(rank + 2)

    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_recommendations(
    recommendations: Mapping[int, Sequence[int]],
    relevant_by_user: Mapping[int, Iterable[int]],
    k_values: Sequence[int] = K_VALUES,
) -> pd.DataFrame:
    """Average Precision, Recall, and NDCG over the evaluated users."""
    rows: list[dict[str, float | int]] = []
    for k in k_values:
        total_precision = 0.0
        total_recall = 0.0
        total_ndcg = 0.0

        for user_id, relevant in relevant_by_user.items():
            recommended = recommendations[user_id]
            total_precision += precision_at_k(recommended, relevant, k)
            total_recall += recall_at_k(recommended, relevant, k)
            total_ndcg += ndcg_at_k(recommended, relevant, k)

        user_count = len(relevant_by_user)
        rows.append(
            {
                "K": k,
                "Precision": total_precision / user_count,
                "Recall": total_recall / user_count,
                "NDCG": total_ndcg / user_count,
            }
        )
    return pd.DataFrame(rows)


def popularity_scores(
    user_artists: pd.DataFrame,
    train: Mapping[int, Iterable[int]],
    use: str = "users",
) -> pd.Series:
    """Rank training artists by training-user count or total training plays."""
    train_pairs_list = []
    for user_id, artists in train.items():
        for artist_id in artists:
            train_pairs_list.append((user_id, artist_id))

    train_pairs = pd.DataFrame(train_pairs_list, columns=["userID", "artistID"])
    scores = train_pairs.merge(user_artists, on=["userID", "artistID"], how="left")
    if use == "users":
        return scores.groupby("artistID")["userID"].nunique().sort_index()
    if use == "plays":
        return scores.groupby("artistID")["weight"].sum().sort_index()
    raise ValueError("use must be either 'users' or 'plays'")


def popularity_recommendations(
    scores: pd.Series,
    train: Mapping[int, Iterable[int]],
    k: int,
) -> Dict[int, list[int]]:
    """Recommend globally popular artists after removing train history."""
    ranked_artists = sorted(
        scores.index.tolist(),
        key=lambda artist: (-scores[artist], artist),
    )
    recommendations = {}

    for user_id, train_artists in train.items():
        seen_artists = set(train_artists)
        user_recommendations = []
        for artist in ranked_artists:
            if artist not in seen_artists:
                user_recommendations.append(artist)
            if len(user_recommendations) == k:
                break
        recommendations[user_id] = user_recommendations

    return recommendations


def unseen_artist_report(
    relevant_by_user: Mapping[int, Iterable[int]],
    training_artists: Iterable[int],
) -> Dict[str, float | int]:
    """Report held-out artists that cannot be candidates for collaborative filtering."""
    training_set = set(training_artists)
    heldout = set()
    for artists in relevant_by_user.values():
        heldout.update(artists)

    unseen = heldout - training_set
    total_interactions = 0
    unseen_interactions = 0
    for artists in relevant_by_user.values():
        artists = set(artists)
        total_interactions += len(artists)
        unseen_interactions += len(artists - training_set)

    return {
        "unique_heldout_artists": len(heldout),
        "unique_unseen_artists": len(unseen),
        "unique_unseen_percentage": 100.0 * len(unseen) / len(heldout)
        if heldout
        else 0.0,
        "heldout_interactions": total_interactions,
        "unseen_interactions": unseen_interactions,
        "unseen_interaction_percentage": 100.0 * unseen_interactions / total_interactions
        if total_interactions
        else 0.0,
    }
