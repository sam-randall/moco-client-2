import joblib

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
import requests
import numpy as np
from typing import Callable

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


class Router:
    def __init__(self, decision: HyperplaneDecisionRealTime, positive_callable: Callable, negative_callable: Callable):
        self.positive_callable = positive_callable
        self.negative_callable = negative_callable
        self.decision = decision
        self.n_early = 0
        self.n_mlp = 0

    def forward(self, x: np.ndarray):
        res = self.decision.forward(x)
        if res: 
            self.n_early += 1
            return self.positive_callable(x)
        else:
            self.n_mlp += 1
            return self.negative_callable(x)


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

    data = response.json()
    W = data['rule']['W']
    b = data['rule']['b']
    t = data['rule']['t']

    hd = HyperplaneDecisionRealTime(np.array(W), np.array(b), t)

    router = Router(hd, lambda x: 0, lambda x: model.predict(x).item())
    return router