import numpy as np
import torch
from tqdm import tqdm

def compute_top_k_accuracy(predictions, targets, k=5):
    correct = 0
    total = 0
    for pred, target in zip(predictions, targets):
        pred = pred.flatten()
        target = target.flatten()
        correct += (pred[:k] == target).sum().item()
        total += 1
    return correct / total if total > 0 else 0.0

def infer_with_test_loader(model, test_loader, device, top_n, movie_tag_tensor, raw2idx, return_top_k=5):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for movie_seq, rating_seq, tag_seq, target, seq_raw in tqdm(test_loader, desc="Inference"):
            movie_seq = movie_seq.to(device)
            rating_seq = rating_seq.to(device)
            tag_seq = tag_seq.to(device)
            target = target.to(device)
            all_movie_embeds = model.get_movie_embeddings()
            candidate_pools = batch_retrieve_candidate_pool(seq_raw, rating_seq, movie_tag_tensor, raw2idx, top_n).to(device)
            seq_embed = model.forward_embedding(movie_seq, rating_seq, tag_seq)
            candidate_embeds = all_movie_embeds[candidate_pools]
            batch_logits = torch.bmm(seq_embed.unsqueeze(1), candidate_embeds.transpose(1, 2)).squeeze(1)
            eq = (candidate_pools == target.unsqueeze(1))
            target_mask = eq.any(dim=1)
            if target_mask.sum() == 0:
                continue
            valid_logits = batch_logits[target_mask]
            valid_eq = eq[target_mask]
            valid_targets = valid_eq.float().argmax(dim=1)
            topk_scores, topk_movies = valid_logits.topk(return_top_k, dim=1, largest=True, sorted=True)
            all_predictions.append(topk_movies.cpu().numpy())
            all_targets.append(valid_targets.cpu().numpy())
    accuracy = compute_top_k_accuracy(np.concatenate(all_predictions), np.concatenate(all_targets), k=return_top_k)
    return all_predictions, all_targets, accuracy
