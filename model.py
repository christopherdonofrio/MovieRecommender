import pandas as pd
from sklearn.model_selection import train_test_split




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


#print(ratings.head())
#print(movies.head())


train_df, test_df = train_test_split(
    ratingsToSplit,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

#print("Train size:", len(train_df))
#print("Test size:", len(test_df))


# Dataset and DataLoad, break dataframes into batches
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math

# create tensors of all userids, movieids, and ratings
class ratingsDataset(Dataset):

    def __init__(self, dataframe):
        #data loading
        self.users = torch.tensor(
            dataframe["user_idx"].values,
            dtype=torch.long
        )

        self.movies = torch.tensor(
            dataframe["movie_idx"].values,
            dtype=torch.long
        )

        self.ratings = torch.tensor(
            dataframe["rating"].values,
            dtype=torch.float32
        )

    def __getitem__(self, index):
        return self.users[index], self.movies[index], self.ratings[index]

    def __len__(self):
        return len(self.ratings)
    

train_dataset = ratingsDataset(train_df)
test_dataset = ratingsDataset(test_df)

trainDataLoader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=0)
testDataLoader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=0)



users, movies, ratings_batch = next(iter(trainDataLoader))



import torch.nn as nn

num_users = ratings["user_idx"].nunique()
num_movies = ratings["movie_idx"].nunique()

class MovieRecommender(nn.Module):
    def __init__(self, num_users, num_movies, global_mean, embedding_dim=20):
        super().__init__()

        # create table of num_users/movies * dimension
        # each unique user and movie gets a vector of size 50
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)

        self.user_biases = nn.Embedding(num_users, 1)
        self.movie_biases = nn.Embedding(num_movies, 1)

        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.movie_biases.weight)

        self.global_bias = nn.Parameter(torch.tensor(global_mean, dtype=torch.float32))

        

    def forward(self, users, movies):
        # users is a tensor of 256 shuffled users
        # movies is a tensor of 256 shuffled movies
        # where user[0] matches movie[0], and ratings[0] is user[0]'s rating of movie[0]


        # vectors of batch size * dimension
        # 256 users, each with a unique vector of 50 numbers
        user_vecs = self.user_embeddings(users)

        movie_vecs = self.movie_embeddings(movies)

        # dot product user and movie vectors and add them, creating a prediction value
        user_bias = self.user_biases(users).squeeze()
        movie_bias = self.movie_biases(movies).squeeze()

        


        # create raw score of dot product of user and movie vecs + biases
        raw_scores = (user_vecs * movie_vecs).sum(dim=1)
        raw_scores = raw_scores + user_bias + movie_bias

        predictions = raw_scores + user_bias + movie_bias + self.global_bias
        predictions = torch.clamp(predictions, 0.5, 5.0)
        
        return predictions


global_mean = ratings["rating"].mean()


model = MovieRecommender(num_users, num_movies, global_mean)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)

epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for users, movies, ratings_batch in trainDataLoader:

        # run the forward pass
        predictions = model(users, movies)

        # return loss 
        loss = loss_fn(predictions, ratings_batch)

        # zero all gradients before backward pass
        optimizer.zero_grad()
        # backpropogate and calculate gradient changes
        loss.backward()
        # update parameters
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(trainDataLoader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")



model.eval()

total_test_loss = 0

with torch.no_grad():
    for users, movies, ratings_batch in testDataLoader:
        predictions = model(users, movies)
        loss = loss_fn(predictions, ratings_batch)
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(testDataLoader)
rmse = avg_test_loss ** 0.5

print(f"Test MSE: {avg_test_loss}")
print(f"Test RMSE: {rmse}")
