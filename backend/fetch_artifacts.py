"""
Downloads the runtime data/model files that are intentionally excluded from
git but that recommender.py needs to start up:
the trained model checkpoint, the cleaned movies CSV, the cached scalar
metadata, and the raw tags CSV.

Run once before starting the server (see Dockerfile). Files already present
on disk (ie. in local dev, where they exist from running train.py directly)
are left alone, this only fills in what's missing.

Expects the four files to be attached as assets to a single GitHub Release,
named exactly as they appear in ARTIFACTS below. Configure which release via
the GITHUB_REPO / RELEASE_TAG environment variables.
"""

import os
import pathlib
import urllib.request

GITHUB_REPO = os.environ.get("GITHUB_REPO", "christopherdonofrio/MovieRecommender")
RELEASE_TAG = os.environ.get("RELEASE_TAG", "model-artifacts-v1")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

ARTIFACTS = [
    ("models/movieRecommenderModel.pt", "movieRecommenderModel.pt"),
    ("data/processed/movies_clean.csv", "movies_clean.csv"),
    ("data/processed/meta.json", "meta.json"),
    ("data/raw/tags.csv", "tags.csv"),
]


def fetch_artifacts():
    base_url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

    for relative_path, asset_name in ARTIFACTS:
        destination = REPO_ROOT / relative_path

        if destination.exists():
            print(f"[fetch_artifacts] {relative_path} already present, skipping")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/{asset_name}"

        print(f"[fetch_artifacts] downloading {relative_path} from {url}")
        urllib.request.urlretrieve(url, destination)
        print(f"[fetch_artifacts] done: {relative_path}")


if __name__ == "__main__":
    fetch_artifacts()
