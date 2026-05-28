# Movie Recommender Model Using PyTorch

## Overview

This is a movie recommender project using PyTorch that aims to predict a user's ratings for an unwatched movie on a 5-star scale. This value will then be used to recommend new movies to watch based on their previous taste in movies, as well as the learned behaviors of thousands of movie reviews (from the MovieLens database) that the model was trained on. I am working on implementing a full machine learning pipeline, including:

- Processing/cleaning MovieLens database
- Contiguously indexing MovieLens users and movie data
- Using PyTorch Dataset and Dataloader to create tensors of corresponding user, movie, and rating data
- Dividing data into a training and test set
- Batching these tensors
- Training a model using backpropagation
- Evaluating and minimizing loss
