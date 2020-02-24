import numpy as np


class PCA():

    """
    PCA projection
    PCA(n_components)

    fit(X, standardized=True) : fits model

    transform(X_scaled) : projects design matrix into PCs
        return projected matrix

    fit_transform(X, standardized=True) : fit + transform

    explained_variance_ratio():
        return array of size n_components
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, standardized=True):
        self.n, self.p = X.shape
        self.X = X
        if standardized:
            self.X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        else:
            self.X_scaled = X

        if self.n_components >= self.p:
            raise ValueError('n_components shoulb be < to p')

        Sigma = np.dot(self.X_scaled.T, self.X_scaled) / self.n
        # Compute eigenvalues and eigenvectors of Sigma
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        # Take the n_components eigenvectors with the biggest eigenvalues
        self.Uq = eigenvectors[:, :self.n_components]

    def transform(self, X_scaled):

        self.X_projected = X_scaled.dot(self.Uq)
        return self.X_projected

    def fit_transform(self, X, standardized=True):
        self.fit(X, standardized)
        return self.transform(self.X_scaled)

    def explained_variance_ratio(self):

        tot_var = np.var(self.X_scaled, axis=0).sum()
        explained_variance = np.var(self.X_projected, axis=0) / tot_var

        return explained_variance
