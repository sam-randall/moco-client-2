import joblib

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
import requests
import numpy as np
from typing import Callable
import json

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
    def __init__(self, positive_class, negative_path):
        self.positive_path = positive_class
        self.negative_path = negative_path

def analyze(model, dataset):
    np.save("dataset.npy", dataset)
    joblib.dump(model, "model.joblib")

    URL = 'moco-vywr.onrender.com'
    LOCAL_URL = 'localhost:8000'

    response = requests.post(
        f"http://{URL}/analyze/",
        files={
            "model_file": open("model.joblib", "rb"),
            "dataset": open("dataset.npy", "rb"),
        },
    )

    if response.status_code == 200:

        data = response.json()
        W = data['rule']['W']
        b = data['rule']['b']
        t = data['rule']['t']

        hd = HyperplaneDecisionRealTime(np.array(W), np.array(b), t)

        router = Router(hd, lambda x: 0, lambda x: model.predict(x).item())

        positive_path = PathConfig("class", activation=0)
        negative_path = PathConfig("default", activation=lambda x: model.predict(x).item())
        router_config = RouterConfig(positive_path, negative_path)

        router = Router(hd, router_config.positive_path, router_config.negative_path)
        return router
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

    