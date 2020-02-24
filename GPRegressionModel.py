import numpy as np


class GPRegression():
    """Gaussian Process Regression Model"""

    def __init__(self, kernel='gaussian', amplitude=1, length=1, noise_std=1, **kwargs):
        self.amplitude = float(amplitude)
        self.length = float(length)
        self.noise_std = float(noise_std)
        if kernel == 'gaussian':
            self.kernel_func = GPRegression.gaussian_kernel(
                self.amplitude, self.length)
        elif kernel == 'matern':
            self.kernel_func = GPRegression.matern_kernel(
                self.amplitude, self.length)
        elif kernel == 'periodic':
            self.periodicity = float(kwargs['periodicity'])
            self.kernel_func = GPRegression.periodic_kernel(
                self.amplitude, self.length, self.periodicity)
        else:
            raise ValueError(
                'kernel value shoulb be either gaussian, matern or periodic')

    def fit(self, X, y):

        self.n, self.p = X.shape
        self.X_train = X
        self.y_train = y
        self.K = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                self.K[i, j] = self.kernel_func(X[i], X[j])

        self.S_inv = np.linalg.solve(
            (self.K + (self.noise_std**2)), np.eye(self.n))

    def predict(self, X):
        """return expected predictive posterior"""
        n_test = len(X)
        y_mean_pred = np.zeros(n_test)
        for i in range(n_test):
            Kxnew_X = self.kernel_new_vector(X[i])
            y_mean_pred[i] = np.dot(
                Kxnew_X.T, np.dot(self.S_inv, self.y_train))

        return y_mean_pred

    def kernel_new_vector(self, x_new):
        """Compute k(., X)"""

        Kxnew_X = np.zeros(self.n)
        for i in range(self.n):
            Kxnew_X[i] = self.kernel_func(x_new, self.X_train[i])

        return Kxnew_X

    @staticmethod
    def gaussian_kernel(amplitude, length):
        """Return a gaussian kernel function with amplitude and length"""

        def kernel_fun(xi, xj):
            norm_sqr = np.sum((xi - xj)**2)
            return (amplitude**2) * np.exp(- norm_sqr / (length**2))

        return kernel_fun

    @staticmethod
    def matern_kernel(amplitude, length):

        def kernel_fun(xi, xj):
            norm = (np.sqrt(3) / length) * np.sqrt(np.sum((xi - xj)**2))
            return (amplitude**2) * (1 + norm) * np.exp(-norm)

        return kernel_fun

    @staticmethod
    def periodic_kernel(amplitude, length, periodicity):

        def kernel_fun(xi, xj):

            sin_term = np.sin(
                periodicity * np.sum((xi - xj)**2) / (2 * np.pi)) ** 2
            return (amplitude**2) * np.exp(-2 * sin_term / (length**2))

        return kernel_fun
