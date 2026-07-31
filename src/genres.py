import torch

GENRE_SEPARATOR = "|" # this is how they're stored in the CSV
NO_GENRES_TOKEN = "(no genres listed)"


def build_genre_vocab(movies_df):
    # Create sorted list of every distinct genre token in the db
    tokens = set()

    for genres_str in movies_df["genres"].fillna(NO_GENRES_TOKEN):
        tokens.update(genres_str.split(GENRE_SEPARATOR))

    return sorted(tokens)


def build_movie_genre_matrix(movies_df, genre_vocab):
    # [num_movies, num_genres] binary presence matrix indexed by movie_idx.
    genre_to_col = {genre: i for i, genre in enumerate(genre_vocab)}
    num_movies = movies_df["movie_idx"].nunique()
    num_genres = len(genre_vocab)

    matrix = torch.zeros((num_movies, num_genres), dtype=torch.float32)

    for movie_idx, genres_str in zip(
        movies_df["movie_idx"], movies_df["genres"].fillna(NO_GENRES_TOKEN)
    ):
        cols = [genre_to_col[token] for token in genres_str.split(GENRE_SEPARATOR) if token in genre_to_col]
        matrix[movie_idx, cols] = 1.0

    return matrix
