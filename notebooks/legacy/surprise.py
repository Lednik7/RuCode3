import pandas as pd
import numpy as np
from surprise import SVD, SVDpp
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Reader


# anime = pd.read_csv("anime.csv").replace("Unknown", np.nan)
print("Reading files...")
ratings = pd.read_csv("user_ratings.csv", dtype=np.int32)
ratings = ratings.loc[ratings["rating"] != 0]

# We'll use the famous SVD++ algorithm.
algo = SVDpp(verbose=True)

print("Loading dataset...")
data = Dataset.load_from_df(ratings[["user_id", "anime_id", "rating"]], Reader(rating_scale=(1, 10)))

# sample random trainset and testset
# test set is made of 10% of the ratings.
trainset, testset = train_test_split(data, test_size=.1)

# Train the algorithm on the trainset, and predict ratings for the testset
print("Training...")
algo.fit(trainset)

# Testing
sample = pd.read_csv("sample_submission.csv")
sample_new = [[int(i) for i in id.split()] + [1] for id, rat in sample.values]
# test_set = pd.DataFrame(sample_new)
test_preds = algo.test(sample_new)
sample["rating"] = pd.Series([x.est for x in test_preds])
sample.to_csv("sub19.csv", index=None)

# Evaluate
print("Evaluating...")
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)
