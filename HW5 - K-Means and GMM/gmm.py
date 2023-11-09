import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, clusters, dataset):
        self.dataset = dataset
        self.clusters = clusters
        self.phi = [1 / clusters] * clusters
        self.means = [dataset[np.random.randint(len(self.dataset))] for i in range(clusters)]
        self.cov = [np.cov(dataset.T) for _ in range(clusters)]
        self.converged = False

        iter = 0
        while not self.converged:
            iter += 1
            weights = self.e_step()
            self.m_step(weights)

    def gaussian(self, j):
        a = (1 / np.sqrt(2 * self.cov[j] * np.pi))
        prob = []
        for x in range(len(self.dataset)):
            b = np.exp(-((self.dataset[x] - self.means[j])**2) / (2 * self.cov[j]))
            prob.append(a * b)
        return prob

    def e_step(self):
        likelihood = np.zeros((self.dataset.shape[0], self.clusters))
        for j in range(self.clusters):
            likelihood[:, j] = multivariate_normal(mean=self.means[j], cov=self.cov[j]).pdf(self.dataset)
        weights = self.phi * likelihood
        weights /= weights.sum(axis=1)[:, np.newaxis]
        return weights

    def m_step(self, weights):
        prev_phi = self.phi
        self.phi = weights.mean(axis=0)

        if np.sum(np.abs(self.phi - prev_phi)) <= 0.01:
            self.converged = True

        for j in range(self.clusters):
            weight = weights[:, [j]]
            total_weight = weight.sum()
            self.means[j] = (weight * self.dataset).sum(axis=0) / total_weight
            self.cov[j] = np.cov(self.dataset.T, aweights=(weight / total_weight).flatten())

    def accuracy(self):
        max_accuracy = 0
        weights = self.e_step()
        predictions = np.argmax(weights, axis=1)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i != j and j != k and i != k:
                        count = 0
                        for idx, pred in enumerate(predictions):
                            if idx < 100 and pred == i:
                                count += 1
                            elif 100 <= idx < 200 and pred == j:
                                count += 1
                            elif 200 <= idx < 300 and pred == k:
                                count += 1
                        acc = count / 300
                        if max_accuracy < acc:
                            max_accuracy = acc
        return max_accuracy

    def log_likelihood(self):
        print(np.sum(-np.log(self.e_step())))
        return np.sum(-np.log(self.e_step()))


