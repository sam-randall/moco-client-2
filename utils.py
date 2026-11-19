import joblib

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
import requests
import numpy as np
from typing import Callable
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass

class HyperplaneDecisionRealTime:
    def __init__(self, W, b, t):
        self.W = W
        self.b = b
        self.t = t

    def forward(self, x: np.ndarray):
        # Assert batch size = 1.
        assert x.shape[0] == 1

        res = x @ self.W + self.b > self.t
        return res.item()

    @staticmethod
    def load(path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        
        return HyperplaneDecisionRealTime(
            np.array(data["W"]),
            np.array(data['b']),
            np.array(data['t'])
        )


    def save(self, path):
        d = {
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            't': self.t
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)


class PathConfig:
    def __init__(self, path_type: str, activation = None):
        self.path_type = path_type
        self.activation = activation

    def forward(self, x: np.ndarray):
        if self.path_type == "class":
            return self.activation
        elif self.path_type == "default":
            return self.activation(x)
        else:
            raise ValueError(f"Unknown path type: {self.path_type}")

class RouterConfig:
    def __init__(self, positive_path, negative_path):
        self.positive_path = positive_path
        self.negative_path = negative_path

class Router:
    def __init__(self, decision: HyperplaneDecisionRealTime, positive_callable: PathConfig, negative_callable: PathConfig):
        self.positive_callable = positive_callable
        self.negative_callable = negative_callable
        self.decision = decision
        self.n_early = 0
        self.n_mlp = 0

    def forward(self, x: np.ndarray):
        res = self.decision.forward(x)
        if res: 
            self.n_early += 1
            return self.positive_callable.forward(x)
        else:
            self.n_mlp += 1
            return self.negative_callable.forward(x)



def analyze(dataset, predictions):
    np.save("dataset.npy", dataset)
    np.save("predictions.npy", predictions)

    URL = 'moco-vywr.onrender.com'
    LOCAL_URL = 'localhost:8000'

    response = requests.post(
        f"http://{URL}/get-all-rules/",
        files={
            "dataset": open("dataset.npy", "rb"),
            "predictions": open("predictions.npy", "rb"),
        },
    )

    if response.status_code == 200:

        data = response.json()
        return data
    else:
        print("failed", response)
        return None



def optimize(model, dataset) -> Router:
    np.save("dataset.npy", dataset)
    joblib.dump(model, "model.joblib")

    URL = 'moco-vywr.onrender.com'
    LOCAL_URL = 'localhost:8000'

    response = requests.post(
        f"http://{URL}/get-sufficient-rule/",
        files={
            "model_file": open("model.joblib", "rb"),
            "dataset": open("dataset.npy", "rb"),
        },
    )

    if response.status_code == 200:

        print("Response", response)
        print("response status code", response.status_code)

        data = response.json()
        W = data['rule']['W']
        b = data['rule']['b']
        t = data['rule']['t']

        hd = HyperplaneDecisionRealTime(np.array(W), np.array(b), t)

        router = Router(hd, lambda x: 0, lambda x: model.predict(x).item())
        return router
    else:
        print("failed", response)
        return None

@dataclass
class Result:
    threshold: float
    group_size: float
    ratio: float
    true_count: int

class HighPrecisionBinaryClassifier:
    def __init__(self, min_precision: float, estimator):
        self.min_precision = min_precision
        self.estimator = estimator
        self.predict_method = None
        self.threshold = None

    @staticmethod
    def scan_for_first_threshold_achieving_min_precision(probabilities: np.ndarray, y: np.ndarray):
        ts = np.linspace(probabilities.min(), probabilities.max(), 100)
        counts = []
        rows = []
        first_threshold = None
        for t in ts:
            mask = probabilities > t
            if mask.sum() == 0:
                continue
            y_counts = pd.Series(y[mask]).value_counts().to_dict()
            true_count = y_counts.get(1, 0)
            
            ratio = true_count / mask.sum()

            rows.append({
                "true_count": true_count,
                "ratio": ratio,
                "threshold": t
            })

            if first_threshold is None and ratio > self.min_precision:
                first_threshold = Result(t, mask.sum(), ratio, true_count)
                

        return first_threshold

    def fit(self, X: np.ndarray, y: np.ndarray):
        model = self.estimator
        model.fit(X, y)
        probabilities = model.predict_proba(X)[:, 1]
        ts = np.linspace(probabilities.min(), probabilities.max(), 100)
        counts = []
        rows = []
        first_threshold = None
        for t in ts:
            mask = probabilities > t
            if mask.sum() == 0:
                continue
            y_counts = pd.Series(y[mask]).value_counts().to_dict()
            true_count = y_counts.get(1, 0)
            
            ratio = true_count / mask.sum()

            rows.append({
                "true_count": true_count,
                "ratio": ratio,
                "threshold": t
            })

            if first_threshold is None and ratio > self.min_precision:
                first_threshold = Result(t, mask.sum(), ratio, true_count)
                self.threshold = first_threshold.threshold
                
    def predict(self, X: np.ndarray):
        out = self.estimator.predict_proba(X)[:, 1]
        return out > self.threshold


class Workspace:

    def __init__(self):
        self.dataset = None
        self.predictions = None

    def add_dataset(self, X: np.ndarray):
        self.dataset = X

    def add_predictions(self, predictions: np.ndarray):
        self.predictions = predictions

    def fit_logistic_regression_proxy(self):
        

        model = LogisticRegression()

        X_train, X_test, y_train, y_test = train_test_split(self.dataset, self.predictions)
        model.fit(X_train, y_train)
        return model


    def fit_high_precision_binary_classifier(self, min_precision: float, X_train: np.ndarray, y_train: np.ndarray):

        model = LogisticRegression()
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_train)[:, 1]
        ts = np.linspace(probabilities.min(), probabilities.max(), 100)
        counts = []
        rows = []
        first_threshold = None
        for t in ts:
            mask = probabilities > t
            y_counts = pd.Series(y_train[mask]).value_counts().to_dict()
            true_count = y_counts.get(1, 0)
            ratio = true_count / mask.sum()

            rows.append({
                "true_count": true_count,
                "ratio": ratio,
                "threshold": t
            })

            if first_threshold is None and ratio > min_precision:
                first_threshold = Result(t, mask.sum(), ratio, true_count)
                
        return Workspace.build_model(model, first_threshold)


    @staticmethod
    def build_model(model: LogisticRegression, result: Result):
        def forward(x: np.ndarray):
            out = model.predict_proba(x)[:, 1]
            return out > result.threshold
        return forward

class CascadedBinaryClassifier:
    def __init__(self, router_estimator, base_model):
        self.router_estimator = router_estimator
        self.base_model = base_model

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.base_model.fit(x, y)
        probs = self.base_model.predict_proba(x)[:, 1]
        binary_classifier = HighPrecisionBinaryClassifier(0.95)
        first_threshold_result = HighPrecisionBinaryClassifier.scan_for_first_threshold_achieving_min_precision(probs, y)
        self.router_estimator = HighPrecisionBinaryClassifier

    def predict(self, x: np.ndarray):
        pass





def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
    model.fit(X, y == 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y == 0)
    high_precision_model = HighPrecisionBinaryClassifier(min_precision=0.9, estimator= LogisticRegression())
    high_precision_model.fit(X_train, y_train)
    test_pred = high_precision_model.predict(X_test)
    print(pd.Series(y_test[test_pred]).value_counts().to_dict())
    
    data = analyze(X, model.predict(X))

    print("analyze data", data)

if __name__ == "__main__":
    main()
