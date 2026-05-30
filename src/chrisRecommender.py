import pandas as pd
from processData import loadAndCleanData
import torch
from model import MovieRecommender
import torch.nn as nn
from helpers import normalizeTitle


torch.manual_seed(42)

train_df, test_df, ratings, movies, global_mean, num_users, num_movies = loadAndCleanData();


myRatings = pd.read_csv("../data/ChrisRatings.csv")

myRatings['Name'] = myRatings["Name"] + " (" + myRatings["Year"].astype(str) + ")"
  
# map my movies to movieidxs
# create a tensor of my userId, and movie movieixs
# create prediction based off that 

movieLens = pd.read_csv("../data/movies_clean.csv")

myRatings["matchTitle"] = myRatings["Name"].apply(normalizeTitle)
movieLens["matchTitle"] = movieLens["title"].apply(normalizeTitle)

#merge movies in myRatings and and movieLens with the same name
# essentially add the correct movie_idx to the myRatings table
myRatings = myRatings.merge(
    movieLens[["matchTitle", "movie_idx"]],
    on="matchTitle",
    how="left"
)

# remove movies that aren't in movieLens database
myRatings = myRatings.dropna(subset=["movie_idx"])


myRatings['movie_idx'] = myRatings['movie_idx'].astype(int)
# drop unneccesary columns
myRatings = myRatings.drop(columns=['Date', 'Name', 'Year', 'Letterboxd URI'])
#print(myRatings.head())

myRatings['ratingCentered'] = myRatings['Rating'] - global_mean


# create tensors of my movie ids and myRatings (centered)
myMovieTensor = torch.tensor(myRatings['movie_idx'].values, dtype=torch.long)
myRatingsTensor = torch.tensor(myRatings['ratingCentered'].values, dtype=torch.float32)



#print(myMovieTensor, myRatingsTensor)



watchedMovies = set(myRatings["movie_idx"].unique())
allMovies = set(movieLens["movie_idx"].unique())

unwatchedMovies = sorted(allMovies - watchedMovies)




model = MovieRecommender(num_users, num_movies, global_mean)
loss_fn = nn.MSELoss()

model.load_state_dict(torch.load("../models/movieRecommenderModel.pt"))
model.eval()

for param in model.parameters():
    param.requires_grad = False

embedding_dim = model.user_embeddings.embedding_dim

tempUserEmbedding = torch.nn.Parameter(
    torch.randn(embedding_dim)
)

tempUserBias = torch.nn.Parameter(
    torch.zeros(1)
)

optimizer = torch.optim.Adam(
    [tempUserEmbedding, tempUserBias],
    lr=0.01
)

temp_epochs = 300

for epoch in range(temp_epochs):
    movieVectors = model.movie_embeddings(myMovieTensor)
    movieBiases = model.movie_biases(myMovieTensor).squeeze()

    predictions = (movieVectors * tempUserEmbedding).sum(dim=1)
    predictions = predictions + movieBiases + tempUserBias

    loss = loss_fn(predictions, myRatingsTensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Temp user epoch {epoch}, Loss: {loss.item()}")



unwatchedMovieTensor = torch.tensor(unwatchedMovies, dtype=torch.long)

with torch.no_grad():
    unseenMovieVectors = model.movie_embeddings(unwatchedMovieTensor)
    unseenMovieBiases = model.movie_biases(unwatchedMovieTensor).squeeze()

    unseenPredictions = (unseenMovieVectors * tempUserEmbedding).sum(dim=1)
    unseenPredictions = unseenPredictions + unseenMovieBiases + tempUserBias
    unseenPredictions = unseenPredictions + global_mean
    unseenPredictions = torch.clamp(unseenPredictions, 0.5, 5.0)

movieScores = list(zip(unwatchedMovies, unseenPredictions.tolist()))
movieScores.sort(key=lambda x: x[1], reverse=True)

top10 = movieScores[:10]
for movie_idx, score in top10:
    movie_row = movieLens[movieLens["movie_idx"] == movie_idx].iloc[0]
    print(movie_row["title"], round(score, 2))