# Dataset and DataLoad, break dataframes into batches
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
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
            dataframe["ratingCentered"].values,
            dtype=torch.float32
        )

    def __getitem__(self, index):
        return self.users[index], self.movies[index], self.ratings[index]

    def __len__(self):
        return len(self.ratings)
    