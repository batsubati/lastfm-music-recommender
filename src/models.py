"""Simple recommendation models for Stage 3."""

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

try:
    from evaluation import candidate_artists
except ModuleNotFoundError:
    from .evaluation import candidate_artists


def build_item_similarity(train):
    """Build cosine similarities between artists using training data only."""
    # Give every training artist a column number.
    artists_set = set()
    for user_artists in train.values():
        artists_set.update(user_artists)
    artists = sorted(artists_set)
    artist_to_row = {}
    for row, artist in enumerate(artists):
        artist_to_row[artist] = row

    # Make one row per artist and one column per user.
    rows = []
    columns = []
    values = []
    for column, user_artists in enumerate(train.values()):
        for artist in user_artists:
            rows.append(artist_to_row[artist])
            columns.append(column)
            values.append(1)

    artist_user_matrix = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(artists), len(train)),
        dtype=float,
    )

    # Normalize each artist row so a dot product becomes cosine similarity.
    normalized_matrix = normalize(artist_user_matrix, axis=1)
    similarity_matrix = normalized_matrix @ normalized_matrix.T
    return artists, similarity_matrix


def item_cf_recommendations(train, training_artists, artists, similarity_matrix, k=20):
    """Recommend artists using summed item similarities."""
    # Store each artist's row number so we can look up similarities.
    artist_to_row = {}
    for row, artist in enumerate(artists):
        artist_to_row[artist] = row

    recommendations = {}

    for user_id, user_train_artists in train.items():
        # Reuse the shared candidate rule to exclude training artists.
        candidates = candidate_artists(user_id, train, training_artists)
        candidate_rows = []
        for artist in candidates:
            candidate_rows.append(artist_to_row[artist])

        history_rows = []
        for artist in user_train_artists:
            history_rows.append(artist_to_row[artist])

        # Add each candidate's similarity to every artist in the user's history.
        scores = similarity_matrix[candidate_rows][:, history_rows].sum(axis=1)
        scores = np.asarray(scores).ravel()

        # Pair every candidate with its score so we can sort them.
        artist_scores = []
        for artist, score in zip(candidates, scores):
            artist_scores.append((artist, score))

        # Sort by highest score first, then by artist ID for stable ties.
        def sort_by_score(artist_score):
            return -artist_score[1], artist_score[0]

        ranked = sorted(artist_scores, key=sort_by_score)
        recommendations[user_id] = []
        for artist, score in ranked:
            if len(recommendations[user_id]) == k:
                break
            recommendations[user_id].append(artist)

    return recommendations
