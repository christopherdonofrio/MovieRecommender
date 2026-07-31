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

train_df, test_df, ratings, movies, global_mean, num_users, num_movies = (
    load_and_clean_data()
)

tags_df = pd.read_csv("../data/raw/tags.csv")
content_vocab, movie_content_matrix = build_movie_content_matrix(movies, tags_df)
print(f"Using {len(content_vocab)} content features (genres+tags)")

train_dataset = RatingsDataset(train_df)

train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4096,
    shuffle=True,
    num_workers=0,
)

users, movies, ratings_batch = next(iter(train_data_loader))


# create model and test
model = MovieRecommender(num_users, num_movies, global_mean, movie_content_matrix).to(device)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-5,
)

# 30 was tuned for the old small dataset (~80K rows) and the old 3-dim embedding model. 
# The full dataset now has ~8.5x more rows per epoch (more
# gradient steps each), and sample testing showed the content-affinity
# model plateaus by epoch ~12-15 out of 30 so 10 epochs is a better-justified number
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for users, movies, ratings_batch in train_data_loader:
        users = users.to(device)
        movies = movies.to(device)
        ratings_batch = ratings_batch.to(device)

        # run the forward pass
        predictions = model(users, movies)

        # return loss
        loss = loss_fn(predictions, ratings_batch)

        # zero all gradients before backward pass
        optimizer.zero_grad()

        # backpropagate and calculate gradient changes
        loss.backward()

        # update parameters
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_data_loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")


# save model after training
torch.save(model.state_dict(), "../models/movieRecommenderModel.pt")
print("Model saved.")