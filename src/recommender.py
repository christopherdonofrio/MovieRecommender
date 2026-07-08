import pandas as pd
import torch
import torch.nn as nn

from helpers import normalize_title
from model import MovieRecommender


torch.manual_seed(42)

ratings = pd.read_csv("../data/processed/ratings_clean.csv")
movie_lens = pd.read_csv("../data/processed/movies_clean.csv")
movie_lens["matchTitle"] = movie_lens["title"].apply(normalize_title)

global_mean = ratings["rating"].mean()
num_users = ratings["user_idx"].nunique()
num_movies = ratings["movie_idx"].nunique()


def load_model():
    model = MovieRecommender(num_users, num_movies, global_mean)
    model.load_state_dict(
        torch.load("../models/movieRecommenderModel.pt", map_location="cpu")
    )
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


model = load_model()


def load_user_data(file_path):
    my_ratings = pd.read_csv(file_path)
    my_ratings["Name"] = (
        my_ratings["Name"] + " (" + my_ratings["Year"].astype(str) + ")"
    )

    return my_ratings


def match_letterboxd_movielens(file_path):
    my_ratings = load_user_data(file_path)
    my_ratings["matchTitle"] = my_ratings["Name"].apply(normalize_title)

    my_ratings = my_ratings.merge(
        movie_lens[["matchTitle", "movie_idx"]],
        on="matchTitle",
        how="left",
    )

    my_ratings = my_ratings.dropna(subset=["movie_idx"])
    my_ratings["movie_idx"] = my_ratings["movie_idx"].astype(int)

    my_ratings["ratingCentered"] = my_ratings["Rating"] - global_mean

    return my_ratings, movie_lens


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

    embedding_dim = model.user_embeddings.embedding_dim

    temp_user_embedding = torch.nn.Parameter(torch.randn(embedding_dim))
    temp_user_bias = torch.nn.Parameter(torch.zeros(1))

    optimizer = torch.optim.Adam(
        [temp_user_embedding, temp_user_bias],
        lr=0.01,
    )

    loss_fn = nn.MSELoss()

    for _ in range(300):
        movie_vectors = model.movie_embeddings(my_movie_tensor)
        movie_biases = model.movie_biases(my_movie_tensor).squeeze()

        predictions = (
            (movie_vectors * temp_user_embedding).sum(dim=1)
            + movie_biases
            + temp_user_bias
        )

        loss = loss_fn(predictions, my_ratings_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    unwatched_movies = get_unwatched_movies(my_ratings, movie_lens)
    unwatched_movie_tensor = torch.tensor(unwatched_movies, dtype=torch.long)

    with torch.no_grad():
        unseen_movie_vectors = model.movie_embeddings(unwatched_movie_tensor)
        unseen_movie_biases = model.movie_biases(unwatched_movie_tensor).squeeze()

        unseen_predictions = (
            (unseen_movie_vectors * temp_user_embedding).sum(dim=1)
            + unseen_movie_biases
            + temp_user_bias
        )

        unseen_predictions += global_mean
        unseen_predictions = torch.clamp(unseen_predictions, 0.5, 5.0)

    movie_scores = list(zip(unwatched_movies, unseen_predictions.tolist()))
    movie_scores.sort(key=lambda movie_score: movie_score[1], reverse=True)

    top_movies = []

    for movie_idx, score in movie_scores[:top_n]:
        movie_row = movie_lens[movie_lens["movie_idx"] == movie_idx].iloc[0]

        top_movies.append(
            {
                "title": movie_row["title"],
                "score": round(score, 2),
            }
        )

    return top_movies