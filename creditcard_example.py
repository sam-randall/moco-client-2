import pandas as pd
from sklearn.neural_network import MLPClassifier
from utils import optimize, HyperplaneDecisionRealTime, Router
import time
import os
from tqdm import tqdm

import warnings

# Suppress only RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_dataset():
    df = pd.read_csv('/Users/samrandall/Downloads/creditcard.csv')
    X = df[[col for col in df.columns if col.startswith('V')]].to_numpy()
    y = df['Class'].to_numpy()
    return X, y

def main():
    X, y = load_dataset()
    model = MLPClassifier()
    model.fit(X, y)

    json_save_path = "creditcard_hyperplane.json"

    if os.path.exists(json_save_path):
    # if False:
        hd = HyperplaneDecisionRealTime.load(json_save_path)
        router = Router(hd, lambda x: 0, lambda x: model.predict(x).item())
    else:
        router = optimize(model, X)
        router.decision.save(json_save_path)

    for i in range(len(X)):
        model.predict(X[i:i+1])

    start = time.time()
    for i in tqdm(range(len(X))):
        router.forward(X[i:i+1])
    end = time.time()
    early_exit_time = end - start

    for i in range(len(X)):
        model.predict(X[i:i+1])

    start = time.time()
    for i in tqdm(range(len(X))):
        model.predict(X[i:i+1])
    end = time.time()
    original_time = end - start

    print("Original Time", original_time) # 13.354 seconds
    print("Early Exit Time", early_exit_time) # 10.707 seconds



if __name__ == "__main__":
    main()