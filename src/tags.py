import torch


def _clean_tags(tags_df, movies_df):
    # Lowercase/strip tag text and drop tags for movies outside the cleaned dataset
    tags_df = tags_df.copy()
    tags_df["tag"] = tags_df["tag"].astype(str).str.strip().str.lower()

    return tags_df.merge(movies_df[["movieId", "movie_idx"]], on="movieId", how="inner")


def build_tag_vocab(tags_df, movies_df, top_k=300):
    # Top-K tags ranked by number of distinct movies they're applied to
    # (broader coverage is more useful than raw application count).
    merged = _clean_tags(tags_df, movies_df)

    counts = merged.groupby("tag")["movie_idx"].nunique().reset_index(name="count")

    # tie-break. count descending, then alphabetical.
    counts = counts.sort_values(by=["count", "tag"], ascending=[False, True])

    return counts["tag"].head(top_k).tolist()


def build_movie_tag_matrix(movies_df, tags_df, tag_vocab):
    # [num_movies, len(tag_vocab)] binary presence matrix indexed by movie_idx
    # Presence only (1 if movie has given tag, 0 if not)
    tag_to_col = {tag: i for i, tag in enumerate(tag_vocab)}
    num_movies = movies_df["movie_idx"].nunique()
    num_tags = len(tag_vocab)

    matrix = torch.zeros((num_movies, num_tags), dtype=torch.float32)

    merged = _clean_tags(tags_df, movies_df)
    merged = merged[merged["tag"].isin(tag_to_col)]
    merged = merged.drop_duplicates(subset=["movie_idx", "tag"])

    if not merged.empty:
        rows = merged["movie_idx"].to_numpy()
        cols = merged["tag"].map(tag_to_col).to_numpy()
        matrix[rows, cols] = 1.0

    return matrix
