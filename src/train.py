import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_one_epoch_vectorized(model, optimizer, criterion, train_loader, device, top_n, movie_tag_tensor, raw2idx):
    model.train()
    total_loss = 0
    all_movie_embeds = model.get_movie_embeddings()  
    for movie_seq, rating_seq, tag_seq, target, seq_raw in tqdm(train_loader, desc="Train Epoch"):
        movie_seq = movie_seq.to(device)
        rating_seq = rating_seq.to(device) 
        tag_seq = tag_seq.to(device)
        target = target.to(device)
        
        candidate_pools = batch_retrieve_candidate_pool(seq_raw, rating_seq, movie_tag_tensor, raw2idx, top_n).to(device)
        
        optimizer.zero_grad()
        seq_embed = model.forward_embedding(movie_seq, rating_seq, tag_seq)
        candidate_embeds = all_movie_embeds[candidate_pools]
        seq_embed_expanded = seq_embed.unsqueeze(1)
        batch_logits = torch.bmm(seq_embed_expanded, candidate_embeds.transpose(1, 2)).squeeze(1)
        eq = (candidate_pools == target.unsqueeze(1))
        target_mask = eq.any(dim=1)
        if target_mask.sum() == 0:
            continue 
        train_logits = batch_logits[target_mask]
        train_eq = eq[target_mask]
        train_targets = train_eq.float().argmax(dim=1)
        train_logits = train_logits / 0.85  # temperature scaling
        loss = criterion(train_logits, train_targets)
        total_loss += loss.item() * train_logits.size(0)
        loss.backward(retain_graph=True)
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

def validate_vectorized(model, criterion, valid_loader, device, top_n, movie_tag_tensor, raw2idx, top_k=5):
    model.eval()
    total_loss = 0
    total = 0
    correct_topk = 0
    skipped = 0 
    all_movie_embeds = model.get_movie_embeddings()
    with torch.no_grad():
        for movie_seq, rating_seq, tag_seq, target, seq_raw in tqdm(valid_loader, desc="Valid Epoch"):
            movie_seq = movie_seq.to(device)
            rating_seq = rating_seq.to(device)
            tag_seq = tag_seq.to(device)
            target = target.to(device)
            candidate_pools = batch_retrieve_candidate_pool(seq_raw, rating_seq, movie_tag_tensor, raw2idx, top_n).to(device)
            seq_embed = model.forward_embedding(movie_seq, rating_seq, tag_seq)
            candidate_embeds = all_movie_embeds[candidate_pools]
            batch_logits = torch.bmm(seq_embed.unsqueeze(1), candidate_embeds.transpose(1, 2)).squeeze(1)
            eq = (candidate_pools == target.unsqueeze(1))
            target_mask = eq.any(dim=1)
            if target_mask.sum() == 0:
                skipped += target.size(0)
                continue
            valid_logits = batch_logits[target_mask]
            valid_eq = eq[target_mask]
            valid_targets = valid_eq.float().argmax(dim=1)
            loss = criterion(valid_logits, valid_targets)
            total_loss += loss.item() * valid_logits.size(0)
            topk_indices = valid_logits.topk(k=top_k, dim=1, largest=True, sorted=True)[1]
            correct_topk += (topk_indices == valid_targets.unsqueeze(1)).any(dim=1).float().sum().item()
            total += valid_logits.size(0)
    avg_loss = total_loss / total if total > 0 else 0.0
    topk_accuracy = correct_topk / total if total > 0 else 0.0
    print(f"Valid Loss: {avg_loss:.4f} | Top-{top_k} Accuracy: {topk_accuracy * 100:.2f}%")
    return avg_loss, topk_accuracy
    