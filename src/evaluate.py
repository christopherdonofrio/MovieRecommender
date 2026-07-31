from processData import load_and_clean_data
from dataset import RatingsDataset
from model import MovieRecommender
from content_features import build_movie_content_matrix

from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

train_df, test_df, ratings, movies, global_mean, num_users, num_movies = load_and_clean_data()

tags_df = pd.read_csv("../data/raw/tags.csv")
content_vocab, movie_content_matrix = build_movie_content_matrix(movies, tags_df)

test_dataset = RatingsDataset(test_df)
test_data_loader = DataLoader(
    test_dataset,
    batch_size=4096,
    shuffle=False,
    num_workers=0,
)

model = MovieRecommender(num_users, num_movies, global_mean, movie_content_matrix).to(device)
loss_fn = nn.MSELoss()

model.load_state_dict(torch.load("../models/movieRecommenderModel.pt", map_location=device))
model.eval()


total_test_loss = 0

with torch.no_grad():
    for users, movies, ratings_batch in test_data_loader:
        users = users.to(device)
        movies = movies.to(device)
        ratings_batch = ratings_batch.to(device)

        predictions = model(users, movies)
        predictions += global_mean
        predictions = torch.clamp(predictions, 0.5, 5.0)

        real_ratings = ratings_batch + global_mean
        loss = loss_fn(predictions, real_ratings)

        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_data_loader)
rmse = avg_test_loss ** 0.5

print(f"Test MSE: {avg_test_loss}")
print(f"Test RMSE: {rmse}")


baseline_prediction = train_df["rating"].mean()
baseline_mse = ((test_df["rating"] - baseline_prediction) ** 2).mean()
baseline_rmse = baseline_mse ** 0.5

print(f"Baseline RMSE: {baseline_rmse}")