This project uses a cosine-similarity function as retriever and lstm-based reranker to make movie recommendation. Input data should be in the form of sequential user ratings on movies. The model learns to predict the next movie_id based on the user profile (movie_id that user rated, the actual rating, corresponding tags value of the movie). Essentially, it is trained in a supervised manner to make movie recommendations given how 20M ratings' users have done. 

Note: 
1. please refer to the notebook for training details as the repository only has simplified training in main.py. 
2. Change configurations in src/config.py.
3. hyperparameter tuning is not yet set up (commented out)

How to Run
python main.py

Output
Best model weights saved to "results/state.pth"
Evaluation metrics (eval accuracy) printed to console in notebook

Install Dependencies
pip install -r requirements.txt

