# Q: Implement the Perceptron algorithm to model the AND logic gate.
import numpy as np
from sklearn.linear_model import Perceptron

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

clf = Perceptron(max_iter=1000, eta0=0.1, random_state=42)

clf.fit(X, y)

print("Weights:", clf.coef_)
print("Bias:", clf.intercept_)

predictions = clf.predict(X)
print("Predictions:", predictions)
