# Movie Recommender Model Using PyTorch

## Overview

This is a movie recommender project that uses PyTorch to predict a user's rating for an unwatched movie on a 5-star scale using collaborative filtering trained on the MovieLens dataset. A user exports their Letterboxd ratings as a CSV file, uploads it through the frontend, and the backend finds matching ratings with MovieLens, fits a personalized model of that user's taste, and returns the highest-predicted-rated unwatched movies (usually top 10).

Pipeline:

* processing/cleaning the MovieLens database
* contiguous indexing MovieLens users and movies
* using a PyTorch `Dataset` / `DataLoader` to batch user/movies/ratings tensors
* training/test split, training via backpropagation, evaluating loss against a baseline

I ended up making significant changes to my model's architecture midway through this project, so below is a rundown of how each file worked in the original version, how they work now, and why I made the switch.

## How it worked (v1)

**`processData.py`** loads the raw MovieLens `ratings.csv` and `movies.csv`, drops users with fewer than 20 ratings and movies with fewer than 10 (and checks again because dropping movies may drop users below the 20-rating threshold), and assigns a contiguous `user_idx` / `movie_idx` index to all the remaining users/movies so they can be embedded later. It also centers each rating around the average of the dataset (`ratingCentered = rating - global_mean`), so the model is actually learning "how above/below the average is this user's rating of this movie", and splits everything into 80/20 train/test sets.

**`dataset.py`** is just a wrapper around a PyTorch `Dataset` class to load the cleaned dataframe into batches of three parallel tensors (user, movie, rating) using a `DataLoader`.

**`helpers.py`** takes care of the annoying string-matching problem of matching a user's Letterboxd rating with the equivalent MovieLens title — by stripping "The"/"A"/accents/punctuation, taking out release years, recognizing "a.k.a." alternate titles, and formatting things neatly for display.

**`model.py`** is just a standard matrix factorization model. Each user and each movie got an embedding vector (I used `embedding_dim = 3`), and the rating prediction was computed as:

```python
prediction = (user_embedding * movie_embedding) + user_bias + movie_bias
```

The theory here is that the dot product learns "some number of taste axes" on its own. Nobody told it anything about what they meant, gradient descent just figured out directions that explained the ratings best, while the two biases absorbed the "this user tends to give high/low ratings" and "this movie is well-liked/less liked" respectively.

**`train.py`** trained this with Adam for 30 epochs and saved the checkpoint. **`evaluate.py`** loaded the model and evaluated RMSE loss of predictions against a baseline (always predicting the training mean).

**`recommender.py`** was the key file where new predictions were served: since the user with ratings uploaded is a completely new user the model hadn't seen before, it froze the model and trained a new `temp_user_embedding` + `temp_user_bias` from scratch against that person's ratings (300 epochs of Adam), then predicted every rating for the rest of the movies using the same dot product formula.

## How it works now (v2)

`processData.py`, `dataset.py`, and `helpers.py` remained unchanged.

**`genres.py`** builds a `[num_movies, 20]` binary matrix with the "genre" information from `movies.csv` (which includes pipe-separated string `"Adventure|Animation|Comedy"` for each movie).

**`tags.py`** does the same thing for free-text tags from `tags.csv`: something like *"murder,"* *"based on a book,"* *"woman director."* I selected the 300 most common tags (ranked by how many different movies they show up on. ie 10,000 times for one movie isn't as useful as 500 times for 500 movies), then built a binary presence matrix for them.

**`content_features.py`** combines the genres and the tags into one `[num_movies, 320]` vector per movie and normalizes each row (otherwise a movie with 15 tags will overwhelm a movie with 2).

**`model.py`** changed completely. There are no more user/movies embeddings, replaced by a user-specific vector of affinity to those 320 content features:

```python
self.user_content_affinity = nn.Embedding(num_users, 320)  # trainable, zero-initialized
self.register_buffer("movie_content_matrix", movie_content_matrix, persistent = False)  # fixed, not learned

prediction = user_bias + movie_bias + (user_content_affinity(user) * movie_content_matrix[movie])
```

Movie content matrix is registered as a buffer (`persistent = False`) to be copied to GPU and loaded in the evaluation stage but not saved in the model checkpoint — no point saving processed data when it can always be recomputed from the CSVs.

**`train.py`**** / ****`evaluate.py`** create this content matrix and feed it into the model. I also reduced the epoch count from 30 to 10. 30 was a leftover from the original tiny dataset and never revisited; on the current dataset the model stops improving after 3 epochs.

**`recommender.py`** trains `temp_user_content_affinity` instead of `temp_user_embedding` using the same frozen content matrix instead of frozen movies embeddings. The process is the same, same Adam 300 epochs.

I also added one thing that wasn't just a mechanical change. `weight_decay=1e-2` to the Adam optimization during that training. Otherwise fitting 320 free parameters off a person's ~100-300 ratings will quickly overfit. I tested this directly using a fake user who rated only horror movies highly, and without the regularization it recommended 0 horror movies.

Finally, I had to retrain the model from scratch as the old checkpoint couldn't be upgraded and parameter names didn't match anymore.

## Why I changed it

I found out that the recommendations for two users who exported their ratings (me and my friend) CSVs were almost identical. Same top movie and strong overlap further down the list. This clearly isn't what a "personalized" recommender should do, so I ran a bunch of tests to figure out why the model was biased towards the same few movies.

I ran several experiments: training the same model with different embedding dimensions (3, 16, 32), training a variant with the dot product removed (leaving the biases only), training with regularization disabled. All experiments converged to the same conclusion: the dot product contribution to the predictions wasn't doing anything. A model with only the biases gave the exact same RMSE with the exact same loss curve as the full model. It wasn't necessarily overfitting, rather it wasn't learning anything meaningful. In retrospect, it makes sense. Trying to learn both a user's tastes and a movie characteristics from sparse ratings occurrences is a difficult task, and on this dataset size it wasn't able to extract useful signal from the data, so recommendations were simply "popular movies" for everyone.

The solution was to stop asking the model to discover taste dimensions itself and instead provide observable data for it to work with. In this case I used genres and tags that already existed in MovieLens dataset but weren't being used to create a simpler learning task. The model doesn't need to discover that horror movies exist, it needs to learn how much a particular user likes them.

Of course, it's not a full solution because the RMSE didn't change much, and universally loved movies (Schindler's List, Rear Window, etc.) are still getting recommended to everybody since their general popularity is hard to outweigh by single person's genre preferences. But the actual bug is definitely improved. The two test users who had almost identical recommendations now have noticeably different, genre-appropriate lists. 