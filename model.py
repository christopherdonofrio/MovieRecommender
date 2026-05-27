import pandas as pd


ratings = pd.read_csv("data/ratings.csv")

ratings = ratings[["userId", "movieId", "rating"]]

# clean ratings to users with atleast 20 movies rated
user_counts = ratings["userId"].value_counts()
active_users = user_counts[user_counts >= 20].index
ratings = ratings[ratings["userId"].isin(active_users)]

#clean movies to only those with 10+ ratings
movies = pd.read_csv("data/movies.csv")
ratingCounts = ratings["movieId"].value_counts()
activeMovies = ratingCounts[ratingCounts >= 10].index
ratings = ratings[ratings["movieId"].isin(activeMovies)]
movies = movies[movies["movieId"].isin(activeMovies)]

# re-check users again to make sure they match
# feels inefficient***
user_counts = ratings["userId"].value_counts()
active_users = user_counts[user_counts >= 20].index
ratings = ratings[ratings["userId"].isin(active_users)]

# Create PyTorch indexes
ratings["user_idx"] = ratings["userId"].astype("category").cat.codes
ratings["movie_idx"] = ratings["movieId"].astype("category").cat.codes

# Save cleaned versions
ratings.to_csv("data/ratings_clean.csv", index=False)
movies.to_csv("data/movies_clean.csv", index=False)

# ratings now with only indexes of users, movies, and ratings
# only what we need to 
ratingsToSplit = ratings[["user_idx", "movie_idx", "rating"]]


print(ratings.head())
print(movies.head())



