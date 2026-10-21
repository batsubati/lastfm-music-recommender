import pytest

from src.models import build_item_similarity, item_cf_recommendations


def test_item_similarity_uses_binary_listening():
    train = {1: frozenset({10, 11}), 2: frozenset({10, 11})}
    artists, similarity = build_item_similarity(train)
    artist_rows = {artist: row for row, artist in enumerate(artists)}

    assert similarity[artist_rows[10], artist_rows[11]] == pytest.approx(1)


def test_recommendations_never_include_training_artists():
    train = {1: frozenset({10, 11}), 2: frozenset({10, 12})}
    artists, similarity = build_item_similarity(train)
    recommendations = item_cf_recommendations(
        train,
        {10, 11, 12},
        artists,
        similarity,
        k=20,
    )

    assert not set(recommendations[1]) & train[1]
