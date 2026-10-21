"""Run the current project stage from one command."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from data import (
    build_interaction_matrix,
    download_and_extract,
    load_dataset,
    run_stage1,
)
from evaluation import (
    K_VALUES,
    SEED,
    evaluate_recommendations,
    popularity_recommendations,
    popularity_scores,
    split_per_user,
    unseen_artist_report,
)
from models import build_item_similarity, item_cf_recommendations


def run_stage2(data_dir: Path, results_dir: Path) -> dict:
    """Run the common split and evaluate the popularity baseline."""
    # 1. Load the listening history.
    dataset_dir = download_and_extract(data_dir)
    frames = load_dataset(dataset_dir)
    interactions = frames["user_artists"]

    # 2. Hide some artists from each eligible user for later evaluation.
    splits = split_per_user(interactions)
    training_artists = set().union(*splits.train.values())

    # 3. Recommend artists using only the training data.
    baseline_scores = popularity_scores(interactions, splits.train, use="users")
    recommendations = popularity_recommendations(
        baseline_scores, splits.train, max(K_VALUES)
    )

    # 4. Compare recommendations with the hidden validation and test artists.
    validation_metrics = evaluate_recommendations(
        recommendations, splits.validation, K_VALUES
    )
    test_metrics = evaluate_recommendations(recommendations, splits.test, K_VALUES)

    result = {
        "eligible_users": len(splits.train),
        "excluded_users": len(splits.excluded_users),
        "excluded_user_ids": sorted(splits.excluded_users),
        "train_users": len(splits.train),
        "validation_users": len(splits.validation),
        "test_users": len(splits.test),
        "train_interactions": sum(map(len, splits.train.values())),
        "validation_interactions": sum(map(len, splits.validation.values())),
        "test_interactions": sum(map(len, splits.test.values())),
        "validation_unseen_artist_report": unseen_artist_report(
            splits.validation, training_artists
        ),
        "test_unseen_artist_report": unseen_artist_report(
            splits.test, training_artists
        ),
        "validation_metrics": validation_metrics.to_dict(orient="records"),
        "test_metrics": test_metrics.to_dict(orient="records"),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "stage2_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    validation_metrics.to_csv(results_dir / "popularity_validation_metrics.csv", index=False)
    test_metrics.to_csv(results_dir / "popularity_test_metrics.csv", index=False)
    return result


def run_stage3(data_dir: Path, results_dir: Path) -> None:
    """Compare item-based collaborative filtering with popularity."""
    # Start the timer so the summary includes the Stage 3 runtime.
    started = time.perf_counter()

    # Create the output folder before saving the metrics file.
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load the listening data and create the same deterministic split as Stage 2.
    dataset_dir = download_and_extract(data_dir)
    interactions = load_dataset(dataset_dir)["user_artists"]
    splits = split_per_user(interactions)
    training_artists = set().union(*splits.train.values())

    # Build item similarities from binary training interactions only.
    artists, similarity_matrix = build_item_similarity(splits.train)
    item_recommendations = item_cf_recommendations(
        splits.train,
        training_artists,
        artists,
        similarity_matrix,
        k=max(K_VALUES),
    )
    item_metrics = evaluate_recommendations(
        item_recommendations,
        splits.validation,
        K_VALUES,
    )
    item_metrics.to_csv(results_dir / "item_cf_validation_metrics.csv", index=False)

    # Build the popularity baseline from the same training data.
    popularity_scores_table = popularity_scores(
        interactions,
        splits.train,
        use="users",
    )
    popularity_recommendations_table = popularity_recommendations(
        popularity_scores_table,
        splits.train,
        max(K_VALUES),
    )
    popularity_metrics = evaluate_recommendations(
        popularity_recommendations_table,
        splits.validation,
        K_VALUES,
    )

    # Put both models side by side so their validation results are easy to compare.
    comparison = item_metrics.copy()
    comparison["Model"] = "Item-based CF"
    baseline = popularity_metrics.copy()
    baseline["Model"] = "Popularity baseline"
    comparison = pd.concat([comparison, baseline], ignore_index=True)

    # Put the columns in the requested order and save both models together.
    comparison = comparison[["Model", "K", "Precision", "Recall", "NDCG"]]
    comparison.to_csv(results_dir / "stage3_comparison.csv", index=False)

    # Save the main facts about this Stage 3 run.
    summary = {
        "validation_users": len(splits.validation),
        "K_values": list(K_VALUES),
        "similarity": "cosine",
        "seed": SEED,
        "runtime_seconds": time.perf_counter() - started,
    }
    (results_dir / "stage3_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Print the comparison table so it is visible immediately in the terminal.
    print(comparison.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Last.fm recommender stages.")
    parser.add_argument("--stage", choices=(1, 2, 3), type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.stage == 1:
        print(run_stage1(args.data_dir, args.results_dir))
        return
    if args.stage == 3:
        run_stage3(args.data_dir, args.results_dir)
        return
    started = time.perf_counter()
    result = run_stage2(args.data_dir, args.results_dir)
    print(f"Stage 2 complete in {time.perf_counter() - started:.2f}s")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
