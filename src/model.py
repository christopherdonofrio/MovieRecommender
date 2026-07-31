import torch.nn as nn


class MovieRecommender(nn.Module):
    def __init__(self, num_users, num_movies, global_mean, movie_content_matrix):
        super().__init__()

        num_content_features = movie_content_matrix.shape[1]

        self.user_biases = nn.Embedding(num_users, 1)
        self.movie_biases = nn.Embedding(num_movies, 1)

        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.movie_biases.weight)

        # Trainable user affinity instead of genre+tag content features,
        # replacing the old arbitrary user embeddings
        self.user_content_affinity = nn.Embedding(num_users, num_content_features)
        nn.init.zeros_(self.user_content_affinity.weight)

        # Fixed, non-trainable per-movie content vector. persistent=False
        # keeps it out of the checkpoint since it's easy to rebuild from
        # movies_clean.csv/tags.csv.
        self.register_buffer("movie_content_matrix", movie_content_matrix, persistent=False)

    def forward(self, users, movies):
        user_bias = self.user_biases(users).squeeze()
        movie_bias = self.movie_biases(movies).squeeze()

        user_content_vecs = self.user_content_affinity(users)
        movie_content_vecs = self.movie_content_matrix[movies]

        content_score = (user_content_vecs * movie_content_vecs).sum(dim=1)

        return user_bias + movie_bias + content_score
