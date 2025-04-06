import torch.nn as nn
import torch.nn.functional as F
import torch

class MovieRatingTagLSTM(nn.Module):
    def __init__(self, num_movies, tag_dim, emb_dim=64, hidden_dim=128, lstm_layers=4, dropout=0.3, proj_dim=128):
        super().__init__()
        self.movie_emb = nn.Embedding(num_movies, emb_dim)
        self.rating_fc = nn.Linear(1, emb_dim)
        self.tag_encoder = nn.Sequential(
            nn.Linear(tag_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        input_dim = emb_dim * 3
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=8, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.seq_projection = nn.Linear(512, proj_dim)
        self.movie_projection = nn.Linear(emb_dim, proj_dim)
        
    def forward_embedding(self, movie_seq, rating_seq, tag_seq):
        movie_vec = self.movie_emb(movie_seq)
        rating_vec = self.rating_fc(rating_seq.unsqueeze(-1))
        tag_vec = self.tag_encoder(tag_seq)
        x = torch.cat([movie_vec, rating_vec, tag_vec], dim=-1)
        lstm_out, _ = self.lstm(x)
        attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        attn_output = torch.mean(attn_output, dim=1)
        x = F.relu(self.fc1(attn_output))
        x = F.relu(self.fc2(x))
        seq_embed = self.seq_projection(x)
        seq_embed = F.normalize(seq_embed, p=2, dim=1)
        return seq_embed
    
    def get_movie_embeddings(self):
        with torch.no_grad():
            movie_embeds = self.movie_emb.weight              
            proj_movie_embeds = self.movie_projection(movie_embeds)
            proj_movie_embeds = F.normalize(proj_movie_embeds, p=2, dim=1)
        return proj_movie_embeds
