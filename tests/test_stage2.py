import pandas as pd

from src.evaluation import (
    candidate_artists,
    evaluate_recommendations,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    split_per_user,
)


def test_split_has_no_overlap_and_excludes_short_users():
    frame = pd.DataFrame(
        {
            "userID": [1] * 5 + [2] * 4,
            "artistID": [10, 11, 12, 13, 14, 20, 21, 22, 23],
            "weight": 1,
        }
    )
    splits = split_per_user(frame)
    assert splits.excluded_users == frozenset({2})
    assert len(splits.train[1] | splits.validation[1] | splits.test[1]) == 5
    assert not splits.train[1] & splits.validation[1]
    assert not splits.train[1] & splits.test[1]
    assert not splits.validation[1] & splits.test[1]


def test_metrics_on_tiny_example():
    recommendations = [3, 2, 5]
    relevant = {2, 4}
    assert precision_at_k(recommendations, relevant, 2) == 0.5
    assert recall_at_k(recommendations, relevant, 2) == 0.5
    expected = (1 / 1.584962500721156) / (1 + 1 / 1.584962500721156)
    assert ndcg_at_k(recommendations, relevant, 2) == expected


def test_evaluation_loop_averages_users():
    result = evaluate_recommendations({1: [10, 11]}, {1: {11}}, [2])
    assert result.loc[0, "Precision"] == 0.5
    assert result.loc[0, "Recall"] == 1.0


def test_candidates_exclude_training_items():
    candidates = candidate_artists(1, {1: {10, 11}}, {10, 11, 12, 13})
    assert candidates == [12, 13]
