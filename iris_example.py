from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from utils import optimize
import numpy as np

def main():
    
    X, y = load_iris(return_X_y = True)

    model = MLPClassifier()
    model.fit(X, y)

    router = optimize(model, X)
    out = router.forward(X[0][np.newaxis, :])
    print(out)

if __name__ == "__main__":
    main()