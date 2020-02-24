import numpy as np


class bayesian_linear_model():

    def __init__(self, alpha=1, sigma=1):
        self.alpha = alpha
        self.sigma = sigma

    def compute_Sigma_(self, X):
        return np.linalg.solve((np.dot(X.T, X) + (self.sigma**2) / self.alpha), (self.sigma**2) * np.eye(X.shape[1]))

    def compute_mu_(self, S, X, y):
        return np.dot(S, np.dot(X.T, y)) / (self.sigma**2)

    def fit(self, X, y):
        self.S = self.compute_Sigma_(X)
        self.mu = self.compute_mu_(self.S, X, y)

    def predict(self, X):
        return np.dot(X, self.mu)
