import torch

class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    hyper_trials = 0
    seq_len = 10
    sample_size = 100000
    bs = 128
    emb_dim = 128
    hidden_dim = 256
    lstm_layers = 2
    dropout = 0.2
    epochs = 3
    lr = 1e-3
    wd = 1e-2
    top_k = 10
    temperature = 0.85
    candidate_pool_size = 100
    patience = 4
    save_model_path = "results/state.pth"
