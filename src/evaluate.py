from processData import load_and_clean_data
from dataset import RatingsDataset
from model import MovieRecommender

from torch.utils.data import DataLoader
import torch
import torch.nn as nn


train_df, test_df, ratings, movies, global_mean, num_users, num_movies = load_and_clean_data()

test_dataset = RatingsDataset(test_df)
test_data_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
)

model = MovieRecommender(num_users, num_movies, global_mean)
loss_fn = nn.MSELoss()

model.load_state_dict(torch.load("../models/movieRecommenderModel.pt"))
model.eval()


total_test_loss = 0

with torch.no_grad():
    for users, movies, ratings_batch in test_data_loader:
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