import numpy as np


class KMeans_Clustering():

    """
    KMeans clustering
    K : number of clusters
    init : initialization. 'points' or 'centroids'.
    """

    def __init__(self, K, init='points', max_iter=300):
        self.K = K
        self.init = init
        self.max_iter = 300

    def fit(self, X):
        self.N, self.P = X.shape
        self.X = X

        if self.init == 'centroids':
            rd_init_mean_idx = np.random.randint(0, self.N, size=self.K)
            init_centroids = X[rd_init_mean_idx].T
            # assignment
            self.assigment_step_(X, init_centroids)
            # update
            self.update_step_(X, self.Z)

        elif self.init == 'points':
            rd_init_assign = np.random.randint(0, self.K, size=self.N)
            self.Z = np.zeros((self.K, self.N))
            for i in range(self.N):
                self.Z[rd_init_assign[i], i] = 1
            # update
            self.update_step_(X, self.Z)

        else:
            raise ValueError("init should be either 'points' or 'centroids'")

        self.compute_epsilon_(X, self.Z, self.centroids)

        nb_iter = 0
        assign_cond = True
        epsilon_cond = True
        while nb_iter < self.max_iter and assign_cond and epsilon_cond:
            old_Z = np.copy(self.Z)
            old_epsilon = self.epsilon
            # assignment
            self.assignment_step_(X, self.centroids)
            # update
            self.update_step_(X, self.Z)

            self.compute_epsilon_(X, self.Z, self.centroids)

            if np.sum(np.abs(old_Z - self.Z)) == self.Z.shape[0] * self.Z.shape[1]:
                assign_cond = False

            if old_epsilon == self.epsilon:
                epsilon_cond = False

            nb_iter += 1

    def predict(self, X):
        N = X.shape[0]
        Z = np.zeros((self.K, N))
        for i in range(N):
            kp = np.argmin(np.sum((self.centroids - X[i])**2, axis=1))
            Z[kp, i] = 1
        return np.argmax(Z, axis=0)

    def fit_predict(self, X):
        self.fit(X)
        return np.argmax(self.Z, axis=0)

    def assignment_step_(self, X, centroids):
        self.Z = np.zeros((self.K, self.N))
        for i in range(self.N):
            kp = np.argmin(np.sum((self.centroids - X[i])**2, axis=1))
            self.Z[kp, i] = 1

    def update_step_(self, X, Z):
        self.centroids = Z.dot(X) / np.sum(Z, axis=1).reshape((self.K, 1))

    def compute_epsilon_(self, X, Z, centroids):
        self.epsilon = 0
        for k in range(self.K):
            self.epsilon += np.sum(Z[k, :] *
                                   np.sum((X - centroids[k])**2, axis=1))
