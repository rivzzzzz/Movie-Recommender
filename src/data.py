import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

# Data loading and preprocessing functions
def load_data():
    genome_scores = pd.read_csv("data/genome_scores.csv")
    genome_tags = pd.read_csv("data/genome_tags.csv")
    links = pd.read_csv("data/link.csv")
    movies = pd.read_csv("data/movie.csv")
    rating = pd.read_csv("data/rating.csv")
    tags = pd.read_csv("data/tag.csv")
    # Preprocess and merge your data as required
    return rating, genome_scores, genome_tags, links, movies, tags

def batch_retrieve_candidate_pool(seq_raw_batch, rating_seq_batch, movie_tag_tensor, raw2idx, top_n):
    device = rating_seq_batch.device
    B = len(seq_raw_batch)
    indices = torch.tensor([[raw2idx[movie_id] for movie_id in seq] for seq in seq_raw_batch], device=device)
    seq_tags = movie_tag_tensor[indices]
    ratings = rating_seq_batch.unsqueeze(2)
    weighted_tags = seq_tags * ratings
    user_profiles = weighted_tags.sum(dim=1)
    user_profiles = F.normalize(user_profiles, p=2, dim=1, eps=1e-8)
    movie_tag_norm = F.normalize(movie_tag_tensor, p=2, dim=1, eps=1e-8)
    sims = torch.matmul(user_profiles, movie_tag_norm.t())
    _, candidate_indices = torch.topk(sims, k=top_n, dim=1)
    return candidate_indices

class MovieRatingTagDataset(Dataset):
    def __init__(self, rating_df, movie_tag_features, seq_len=5):
        self.samples = []
        for user_id, group in rating_df.groupby('userId'):
            group = group.sort_values('timestamp')
            movies_enc = group['movieId_enc'].tolist()
            ratings_list = group['rating'].tolist()
            movies_raw = group['movieId'].tolist() 
            for i in range(len(movies_enc) - seq_len):
                seq_movies = movies_enc[i:i+seq_len]
                seq_ratings = ratings_list[i:i+seq_len]
                seq_raw = movies_raw[i:i+seq_len]
                target = movies_enc[i+seq_len]
                if not all(r in movie_tag_features.index for r in seq_raw):
                    continue
                tag_seq = [movie_tag_features.loc[r].values for r in seq_raw]
                self.samples.append((seq_movies, seq_ratings, tag_seq, target, seq_raw))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq_movies, seq_ratings, tag_seq, target, seq_raw = self.samples[idx]
        tag_seq_array = np.array(tag_seq)
        return (
            torch.tensor(seq_movies, dtype=torch.long),   
            torch.tensor(seq_ratings, dtype=torch.float),   
            torch.tensor(tag_seq_array, dtype=torch.float), 
            torch.tensor(target, dtype=torch.long),         
            seq_raw
        )
