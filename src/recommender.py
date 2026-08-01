import json

import pandas as pd
import torch
import torch.nn as nn

from helpers import (
    extract_movie_year,
    format_movie_title,
    get_movie_title_variants,
    normalize_match_title,
)
from model import MovieRecommender
from content_features import build_movie_content_matrix


torch.manual_seed(42)

# import cleaned movies csv for matching (ratings_clean.csv is not loaded here
# it's multi-GB at full dataset scale, and everything this module needs from it
# is the 3 scalars cached in meta.json by processData.py)
movie_lens = pd.read_csv("../data/processed/movies_clean.csv")
tags_df = pd.read_csv("../data/raw/tags.csv")

# match movies on release year and alternative titles (helper fcts)
movie_lens["matchYear"] = movie_lens["title"].apply(extract_movie_year)
movie_lens["matchVariants"] = movie_lens["title"].apply(get_movie_title_variants)

with open("../data/processed/meta.json") as f:
    _meta = json.load(f)

global_mean = _meta["global_mean"]
num_users = _meta["num_users"]
num_movies = _meta["num_movies"]

content_vocab, movie_content_matrix = build_movie_content_matrix(movie_lens, tags_df)
assert len(content_vocab) == 320, (
    f"expected 20 genres + 300 tags = 320 content features, got {len(content_vocab)} "
    "-- underlying data or top_k_tags likely changed"
)


def load_model():
    # Load the trained collaborative filtering model

    model = MovieRecommender(num_users, num_movies, global_mean, movie_content_matrix)
    model.load_state_dict(
        torch.load("../models/movieRecommenderModel.pt", map_location="cpu")
    )
    model.eval()

    # Freeze the learned embeddings while optimizing a temporary embedding for the new user.
    for param in model.parameters():
        param.requires_grad = False

    return model


model = load_model()


def load_user_data(file_path):

    # Read and clean the exported Letterboxd ratings.

    my_ratings = pd.read_csv(file_path)

    my_ratings["Name"] = my_ratings["Name"].astype(str).str.strip()
    my_ratings["Year"] = pd.to_numeric(
        my_ratings["Year"],
        errors="coerce",
    )

    my_ratings = my_ratings.dropna(
        subset=["Name", "Year", "Rating"]
    ).copy()

    my_ratings["Year"] = my_ratings["Year"].astype(int)

    return my_ratings

# Match a Letterboxd movie to the correct MovieLens movie index.
# Prefer an exact release year match and allow a one-year difference to account for dataset inconsistencie.
def find_movielens_movie(user_title, user_year):
    user_key = normalize_match_title(user_title)

    candidates = movie_lens[
        movie_lens["matchVariants"].apply(
            lambda variants: user_key in variants
        )
    ]

    if candidates.empty:
        return None

    exact_year = candidates[candidates["matchYear"] == user_year]

    if len(exact_year) == 1:
        return int(exact_year.iloc[0]["movie_idx"])

    nearby_year = candidates[
        candidates["matchYear"].notna()
        & ((candidates["matchYear"] - user_year).abs() <= 1)
    ]

    if len(nearby_year) == 1:
        return int(nearby_year.iloc[0]["movie_idx"])

    return None

# Convert the user's Letterboxd ratings into MovieLens idxs so they can be used by the trained recommender
def match_letterboxd_movielens(file_path):
    my_ratings = load_user_data(file_path)

    my_ratings["movie_idx"] = my_ratings.apply(
        lambda row: find_movielens_movie(
            row["Name"],
            int(row["Year"]),
        ),
        axis=1,
    )

    my_ratings = my_ratings.dropna(subset=["movie_idx"]).copy()
    my_ratings["movie_idx"] = my_ratings["movie_idx"].astype(int)

    my_ratings["ratingCentered"] = (
        my_ratings["Rating"] - global_mean
    )

    return my_ratings, movie_lens

# return every movie the user has not already rated
def get_unwatched_movies(my_ratings, movie_lens):
    watched_movies = set(my_ratings["movie_idx"].unique())
    all_movies = set(movie_lens["movie_idx"].unique())

    return sorted(all_movies - watched_movies)


def get_recommendations(file_path, top_n=10):
    my_ratings, movie_lens = match_letterboxd_movielens(file_path)

    if my_ratings.empty:
        return []

    my_movie_tensor = torch.tensor(
        my_ratings["movie_idx"].values,
        dtype=torch.long,
    )

    my_ratings_tensor = torch.tensor(
        my_ratings["ratingCentered"].values,
        dtype=torch.float32,
    )

    num_content_features = model.movie_content_matrix.shape[1]

    # content dims are meaningful (genres/tags), so the fit starts
    # at the population-average prediction and only changes as it learns this specific user's real preferences.
    temp_user_content_affinity = torch.nn.Parameter(torch.zeros(num_content_features))
    temp_user_bias = torch.nn.Parameter(torch.zeros(1))

     # Optimize the temporary user content affinity so it reproduces the user's known ratings.
     # Strong weight_decay here: with 320 free parameters fit from typically only a few
     # hundred ratings (or fewer), this is far more underdetermined than the main training
     # run (25M+ ratings) without real regularization it overfits to noise instead of learning genuine genre/tag preference 
    optimizer = torch.optim.Adam(
        [temp_user_content_affinity, temp_user_bias],
        lr=0.01,
        weight_decay=1e-2,
    )

    loss_fn = nn.MSELoss()

    for _ in range(300):
        movie_content_vectors = model.movie_content_matrix[my_movie_tensor]
        movie_biases = model.movie_biases(my_movie_tensor).squeeze()

        predictions = (
            (movie_content_vectors * temp_user_content_affinity).sum(dim=1)
            + movie_biases
            + temp_user_bias
        )

        loss = loss_fn(predictions, my_ratings_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    unwatched_movies = get_unwatched_movies(my_ratings, movie_lens)
    unwatched_movie_tensor = torch.tensor(unwatched_movies, dtype=torch.long)

    # predict ratings for every movie the user has not rated
    with torch.no_grad():
        unseen_movie_content_vectors = model.movie_content_matrix[unwatched_movie_tensor]
        unseen_movie_biases = model.movie_biases(unwatched_movie_tensor).squeeze()

        unseen_predictions = (
            (unseen_movie_content_vectors * temp_user_content_affinity).sum(dim=1)
            + unseen_movie_biases
            + temp_user_bias
        )

        unseen_predictions += global_mean

        # Convert predictions back to the original rating scale
        unseen_predictions = torch.clamp(unseen_predictions, 0.5, 5.0)

    # rank unseen movies from highest predicted rating to lowest
    movie_scores = list(zip(unwatched_movies, unseen_predictions.tolist()))
    movie_scores.sort(key=lambda movie_score: movie_score[1], reverse=True)

    # format the highest-ranked recommendations for the frontend
    top_movies = []

    for movie_idx, score in movie_scores[:top_n]:
        movie_row = movie_lens[movie_lens["movie_idx"] == movie_idx].iloc[0]

        top_movies.append(
            {
                "title": format_movie_title(movie_row["title"]),
                "score": round(score, 2),
            }
        )

    return top_movies
