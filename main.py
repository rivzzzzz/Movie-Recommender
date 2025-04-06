# main.py

from src.config import CFG
from src.data import load_data, preprocess_data, MovieRatingTagDataset, get_movie_tag_tensor_and_raw2idx
from src.model import MovieRatingTagLSTM
from src.train import train_one_epoch_vectorized, validate_vectorized
from src.inference import infer_with_test_loader
from src.utils import set_seed  

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import random
import gc

# REFER TO NOTEBOOK FOR DETAILS, HERE IS ABBREVIATED VERSION OF TRAINING AND INFERENCE
def main():
    # seed everything
    set_seed(CFG.seed)
    
    rating, genome_scores, genome_tags, links, movies, tags = load_data()
    
    rating_df, movie_tag_df, movie_encoder = preprocess_data(rating, genome_scores, genome_tags, links, movies)
    
    # important tables
    movie_tag_tensor, raw2idx = get_movie_tag_tensor_and_raw2idx(movie_tag_df, CFG)
    
    # use subset of rating_df (default set we have 20M, so we randomly slice out a sequence of 1M)
    start_idx = random.randint(0, len(rating_df) - CFG.sample_size)
    sample_rating_df = rating_df.iloc[start_idx:start_idx + CFG.sample_size]
    
    # for simplicity, here we use the whole sample as training; refer to notebook for details in train-test split.
    train_dataset = MovieRatingTagDataset(sample_rating_df, movie_tag_df, seq_len=CFG.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=CFG.bs, shuffle=True)
    
    # valid and test loader if there's any
    # valid_loader = DataLoader(valid_dataset, batch_size=CFG.bs, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=CFG.bs, shuffle=False)
    
    # --- Build the Model ---
    tag_dim = movie_tag_df.shape[1]
    num_movies = len(movie_encoder.classes_)
    model = MovieRatingTagLSTM(
        num_movies=num_movies, 
        tag_dim=tag_dim, 
        emb_dim=CFG.emb_dim, 
        hidden_dim=CFG.hidden_dim, 
        lstm_layers=CFG.lstm_layers,
        dropout=CFG.dropout,
    ).to(CFG.device)
    
    # --- Set up Optimizer, Criterion, and Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-5)
    
    # --- Training Loop ---
    train_losses = [] #, valid_losses, valid_accs = [], [], []
    best_acc = 0
    patience_count = 0
    best_model_state = model.state_dict()
    
    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch+1}/{CFG.epochs}")
        tloss = train_one_epoch_vectorized(model, optimizer, criterion, train_loader, CFG.device, 
                                           CFG.candidate_pool_size, movie_tag_tensor, raw2idx)
        # # If there's validation loader:
        # vloss, vacc = validate_vectorized(model, criterion, valid_loader, CFG.device, CFG.candidate_pool_size, movie_tag_tensor, raw2idx, CFG.top_k)
        # valid_losses.append(vloss)
        # valid_accs.append(vacc)
        
        # Example: Save the best model based on validation accuracy
        # if vacc > best_acc:
        #     best_acc = vacc
        #     best_model_state = model.state_dict()
        #     patience_count = 0
        #     torch.save(best_model_state, CFG.save_model_path)
        #     print("Best model state updated.")
        # else:
        #     patience_count += 1
        #     if patience_count >= CFG.patience:
        #         print("Early stopping triggered")
        #         break
        
        scheduler.step()

    # save trained model
    torch.save(model.state_dict(), CFG.save_model_path)

    # load best model state before inference (if applicable)
    # model.load_state_dict(best_model_state)
    
    # --- Inference ---
    # replace train_loader with test_loader if there's any
    top_n = 100
    return_top_k = 1
    predictions, targets, accuracy = infer_with_test_loader(model, train_loader, CFG.device, top_n, movie_tag_tensor, raw2idx, return_top_k)
    print(f"Top-{return_top_k} Accuracy: {accuracy * 100:.2f}%")
    
    # free memory
    gc.collect()
    
if __name__ == "__main__":
    main()
