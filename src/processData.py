import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean_data():
    ratings = pd.read_csv("../data/raw/ratings.csv")
    movies = pd.read_csv("../data/raw/movies.csv")

    # clean ratings to users with at least 20 movies rated
    user_counts = ratings["userId"].value_counts()
    active_users = user_counts[user_counts >= 20].index
    ratings = ratings[ratings["userId"].isin(active_users)]

    # clean movies to only those with 10+ ratings
    rating_counts = ratings["movieId"].value_counts()
    active_movies = rating_counts[rating_counts >= 10].index
    ratings = ratings[ratings["movieId"].isin(active_movies)]
    movies = movies[movies["movieId"].isin(active_movies)]

    # re-check users again to make sure they match
    # feels inefficient***
    user_counts = ratings["userId"].value_counts()
    active_users = user_counts[user_counts >= 20].index
    ratings = ratings[ratings["userId"].isin(active_users)]

    # Create PyTorch indexes
    ratings["user_idx"] = ratings["userId"].astype("category").cat.codes
    ratings["movie_idx"] = ratings["movieId"].astype("category").cat.codes

    # map back movieidxs to movieIds for recommendation at the end
    movie_mapping = ratings[["movieId", "movie_idx"]].drop_duplicates()
    movies = movies.merge(movie_mapping, on="movieId", how="inner")

    # Save cleaned versions
    ratings.to_csv("../data/processed/ratings_clean.csv", index=False)
    movies.to_csv("../data/processed/movies_clean.csv", index=False)

    # ratings now with only indexes of users, movies, and ratings
    # rather than guessing exact rating, predict how far rating will be from the mean
    global_mean = ratings["rating"].mean()

    ratings["ratingCentered"] = ratings["rating"] - global_mean

    ratings_to_split = ratings[
        ["user_idx", "movie_idx", "rating", "ratingCentered"]
    ]

    # print(ratings.head())
    # print(movies.head())

    train_df, test_df = train_test_split(
        ratings_to_split,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    num_users = ratings["user_idx"].nunique()
    num_movies = ratings["movie_idx"].nunique()

    return train_df, test_df, ratings, movies, global_mean, num_users, num_movies