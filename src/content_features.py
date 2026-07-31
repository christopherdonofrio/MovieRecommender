import torch

from genres import build_genre_vocab, build_movie_genre_matrix
from tags import build_tag_vocab, build_movie_tag_matrix


def build_movie_content_matrix(movies_df, tags_df, top_k_tags=300):
    # Entry point combining genre + top tag features into one
    # per-movie content vector, so every caller builds it the same way.
    genre_vocab = build_genre_vocab(movies_df)
    genre_matrix = build_movie_genre_matrix(movies_df, genre_vocab)

    tag_vocab = build_tag_vocab(tags_df, movies_df, top_k=top_k_tags)
    tag_matrix = build_movie_tag_matrix(movies_df, tags_df, tag_vocab)

    combined_matrix = torch.cat([genre_matrix, tag_matrix], dim=1)

    # L1-normalize so a movie with many active flags doesn't get a
    # proportionally larger raw dot-product contribution than those that lack them
    row_sums = combined_matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
    combined_matrix = combined_matrix / row_sums

    return genre_vocab + tag_vocab, combined_matrix
