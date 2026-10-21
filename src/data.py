"""Download, load, and explore the HetRec 2011 Last.fm dataset.

This module intentionally stops at Stage 1. Splitting and model-specific
processing belong to the later stages of the project.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

DATASET_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
EXPECTED_FILES = (
    "user_artists.dat",
    "artists.dat",
    "tags.dat",
    "user_taggedartists.dat",
)
USER_ARTIST_COLUMNS = ("userID", "artistID", "weight")
ARTIST_COLUMNS = ("id", "name", "url", "pictureURL")
TAG_COLUMNS = ("tagID", "tagValue")
USER_TAG_COLUMNS = ("userID", "artistID", "tagID", "day", "month", "year")


def _extract_archive_safely(archive_path: Path, data_dir: Path) -> None:
    """Extract an archive only when every file stays inside data_dir."""
    # Resolve the destination once so all archive paths are checked consistently.
    destination = data_dir.resolve()

    # Inspect every archive member before extracting anything.
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(
                    f"Unsafe archive path rejected: {member.filename}"
                )

            # Resolve the final path and verify it remains inside data_dir.
            extracted_path = (data_dir / member.filename).resolve()
            try:
                extracted_path.relative_to(destination)
            except ValueError as exc:
                raise RuntimeError(
                    f"Unsafe archive path rejected: {member.filename}"
                ) from exc

        # Extract only after every member passes the safety checks.
        archive.extractall(data_dir)


def download_and_extract(
    data_dir: Path,
    url: str = DATASET_URL,
    timeout_seconds: int = 30,
) -> Path:
    """Download and extract the dataset, returning its extracted directory.

    Existing complete data is reused. A network failure raises a clear
    ``RuntimeError`` with manual-download instructions rather than hiding the
    failure behind a partially populated data directory.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = data_dir
    if all((extracted_dir / filename).exists() for filename in EXPECTED_FILES):
        return extracted_dir

    archive_path = data_dir / "hetrec2011-lastfm-2k.zip"
    try:
        print(f"Downloading dataset from {url}")
        urllib.request.urlretrieve(url, archive_path)
        _extract_archive_safely(archive_path, data_dir)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            "Automatic download failed. Manually download "
            "hetrec2011-lastfm-2k.zip from "
            "https://grouplens.org/datasets/hetrec-2011/, extract it into "
            f"{data_dir}, and rerun the command."
        ) from exc

    if not all((extracted_dir / filename).exists() for filename in EXPECTED_FILES):
        raise RuntimeError(
            f"Extraction completed, but the expected files were not found in "
            f"{extracted_dir}. Check the archive layout and README."
        )
    return extracted_dir


def load_dataset(dataset_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all four raw files using the schemas documented by GroupLens."""
    missing = [name for name in EXPECTED_FILES if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files in {dataset_dir}: {', '.join(missing)}"
        )

    return {
        "user_artists": pd.read_csv(
            dataset_dir / "user_artists.dat",
            sep="\t",
            header=0,
            names=USER_ARTIST_COLUMNS,
        ),
        "artists": pd.read_csv(
            dataset_dir / "artists.dat",
            sep="\t",
            header=0,
            names=ARTIST_COLUMNS,
            encoding="latin-1",
        ),
        "tags": pd.read_csv(
            dataset_dir / "tags.dat",
            sep="\t",
            header=0,
            names=TAG_COLUMNS,
            encoding="latin-1",
        ),
        "user_taggedartists": pd.read_csv(
            dataset_dir / "user_taggedartists.dat",
            sep="\t",
            header=0,
            names=USER_TAG_COLUMNS,
        ),
    }


def build_interaction_matrix(user_artists: pd.DataFrame) -> sparse.csr_matrix:
    """Build a sparse user-by-artist matrix containing raw play counts.

    Rows and columns are indexed by sorted unique IDs. The matrix stores
    implicit play-count confidence data; later evaluation will use binary
    relevance rather than treating counts as ratings.
    """
    user_ids = np.sort(user_artists["userID"].unique())
    artist_ids = np.sort(user_artists["artistID"].unique())
    user_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
    artist_index = pd.Series(np.arange(len(artist_ids)), index=artist_ids)
    rows = user_artists["userID"].map(user_index).to_numpy()
    cols = user_artists["artistID"].map(artist_index).to_numpy()
    values = user_artists["weight"].to_numpy(dtype=np.float64)
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(user_ids), len(artist_ids))
    )


def summarize_dataset(
    frames: Mapping[str, pd.DataFrame],
    interaction_matrix: sparse.csr_matrix,
) -> Dict[str, object]:
    """Return reproducible Stage 1 counts, sparsity, and distributions."""
    interactions = frames["user_artists"]
    user_item_counts = interactions.groupby("userID").size()
    play_counts = interactions["weight"]
    n_users, n_artists = interaction_matrix.shape
    return {
        "users": int(n_users),
        "artists_with_interactions": int(n_artists),
        "artists_metadata_rows": int(len(frames["artists"])),
        "listening_records": int(len(interactions)),
        "tags": int(len(frames["tags"])),
        "user_taggedartist_records": int(len(frames["user_taggedartists"])),
        "sparsity": float(
            1 - interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1])
        ),
        "play_count_min": int(play_counts.min()),
        "play_count_median": float(play_counts.median()),
        "play_count_mean": float(play_counts.mean()),
        "play_count_max": int(play_counts.max()),
        "items_per_user_min": int(user_item_counts.min()),
        "items_per_user_median": float(user_item_counts.median()),
        "items_per_user_mean": float(user_item_counts.mean()),
        "items_per_user_max": int(user_item_counts.max()),
    }


def save_exploration_outputs(
    frames: Mapping[str, pd.DataFrame],
    interaction_matrix: sparse.csr_matrix,
    results_dir: Path,
) -> Dict[str, object]:
    """Save Stage 1 plots, summary text, and cleaning notes."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    interactions = frames["user_artists"]
    summary = summarize_dataset(frames, interaction_matrix)

    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(interactions["weight"]), bins=40, color="#4472c4")
    plt.xlabel("log1p(play count)")
    plt.ylabel("Number of user-artist records")
    plt.title("Distribution of implicit play-count confidence")
    plt.tight_layout()
    plt.savefig(figures_dir / "play_count_distribution.png", dpi=150)
    plt.close()

    items_per_user = interactions.groupby("userID").size()
    plt.figure(figsize=(8, 5))
    plt.hist(items_per_user, bins=40, color="#70ad47")
    plt.xlabel("Distinct artists per user")
    plt.ylabel("Number of users")
    plt.title("Distribution of artists listened to per user")
    plt.tight_layout()
    plt.savefig(figures_dir / "items_per_user_distribution.png", dpi=150)
    plt.close()

    summary_lines = ["Stage 1 dataset summary", "========================", ""]
    summary_lines.extend(f"{key}: {value}" for key, value in summary.items())
    (results_dir / "stage1_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    cleaning_notes = """# Stage 1 cleaning notes

- The raw files are loaded as tab-separated files using the column schemas in
  the GroupLens archive README.
- `user_artists.dat` play counts are retained as implicit confidence values;
  they are not interpreted as star ratings.
- The exploratory interaction matrix is sparse and stores raw play counts.
- No interaction rows are removed in Stage 1. Duplicate keys, missing values,
  and ID coverage are reported below so later stages can make explicit,
  reproducible decisions.
- Train/validation/test splitting and leakage prevention are intentionally
  deferred to Stage 2.

## Basic data checks

"""
    for name, frame in frames.items():
        cleaning_notes += (
            f"- `{name}`: {len(frame)} rows, "
            f"{int(frame.isna().sum().sum())} missing cells, "
            f"{int(frame.duplicated().sum())} duplicate rows.\n"
        )
    cleaning_notes += (
        f"- Interaction matrix shape: {interaction_matrix.shape}; "
        f"stored nonzero entries: {interaction_matrix.nnz}.\n"
    )
    (results_dir / "cleaning_notes.md").write_text(cleaning_notes, encoding="utf-8")
    return summary


def write_environment(results_dir: Path, runtime_seconds: float) -> None:
    """Write the reproducibility metadata required by the project brief."""
    import matplotlib
    import scipy
    import sklearn

    results_dir.mkdir(parents=True, exist_ok=True)
    content = f"""Python: {sys.version.split()[0]}
Platform: {platform.platform()}
Machine: {platform.machine()}
Seed: 42
NumPy: {np.__version__}
Pandas: {pd.__version__}
SciPy: {scipy.__version__}
scikit-learn: {sklearn.__version__}
Matplotlib: {matplotlib.__version__}
Stage 1 runtime_seconds: {runtime_seconds:.3f}
"""
    (results_dir / "environment.txt").write_text(content, encoding="utf-8")


def run_stage1(data_dir: Path, results_dir: Path) -> Dict[str, object]:
    """Download, load, summarize, and plot the raw dataset."""
    started = time.perf_counter()
    dataset_dir = download_and_extract(data_dir)
    frames = load_dataset(dataset_dir)
    interaction_matrix = build_interaction_matrix(frames["user_artists"])
    summary = save_exploration_outputs(frames, interaction_matrix, results_dir)
    write_environment(results_dir, time.perf_counter() - started)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 Last.fm exploration.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    summary = run_stage1(args.data_dir, args.results_dir)
    print("Stage 1 exploration complete.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
