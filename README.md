# Movie Recommendation System

This project uses a **cosine similarity-based retriever** and an **LSTM-based reranker** to make movie recommendations. Input data should be in the form of sequential user ratings on movies. 

The model learns to predict the **next movie_id** based on the user profile — which includes:
- movie_ids that the user has rated,
- the actual rating,
- corresponding tag values of the movie.

It is trained in a **supervised manner** using the [MovieLens 20M dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).

---

## Notes

1. Please refer to the **notebook** for full training details. `main.py` only provides a simplified training pipeline.
2. Change configurations in [`src/config.py`](src/config.py).
3. Hyperparameter tuning is **not yet set up** (currently commented out).

---

## How to Run

```bash
python main.py