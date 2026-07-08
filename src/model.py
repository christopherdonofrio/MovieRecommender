import torch.nn as nn


class MovieRecommender(nn.Module):
    def __init__(self, num_users, num_movies, global_mean, embedding_dim=3):
        super().__init__()

        # create table of num_users/movies * dimension
        # each unique user and movie gets a vector of size 3
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)

        self.user_biases = nn.Embedding(num_users, 1)
        self.movie_biases = nn.Embedding(num_movies, 1)

        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.movie_biases.weight)

    def forward(self, users, movies):
        # users is a tensor of 256 shuffled users
        # movies is a tensor of 256 shuffled movies
        # where user[0] matches movie[0], and ratings[0]
        # is user[0]'s rating of movie[0]

        # vectors of batch size * dimension
        # 256 users, each with a unique vector of 3 numbers
        user_vecs = self.user_embeddings(users)
        movie_vecs = self.movie_embeddings(movies)

        # dot product user and movie vectors and add them,
        # creating a prediction value
        user_bias = self.user_biases(users).squeeze()
        movie_bias = self.movie_biases(movies).squeeze()

        # create raw score of dot product of
        # user and movie vectors + biases
        raw_scores = (user_vecs * movie_vecs).sum(dim=1)

        predictions = raw_scores + user_bias + movie_bias

        return predictions