from processData import load_and_clean_data
from dataset import RatingsDataset
from model import MovieRecommender

from torch.utils.data import DataLoader
import torch
import torch.nn as nn


train_df, test_df, ratings, movies, global_mean, num_users, num_movies = (
    load_and_clean_data()
)

train_dataset = RatingsDataset(train_df)

train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,
)

users, movies, ratings_batch = next(iter(train_data_loader))


# create model and test
model = MovieRecommender(num_users, num_movies, global_mean)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-5,
)

epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for users, movies, ratings_batch in train_data_loader:

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


torch.save(model.state_dict(), "../models/movieRecommenderModel.pt")
print("Model saved.")