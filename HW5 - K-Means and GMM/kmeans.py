import numpy as np
import math


class KMeans:

    def __init__(self, clusters, dataset):
        self.dataset = dataset
        self.centroids = [dataset[np.random.randint(len(self.dataset))] for i in range(clusters)]
        self.converged = False

        iter = 0
        while not self.converged:
            iter += 1
            self.update()

    def closest_centroid(self, x):
        distances = np.linalg.norm(self.centroids - x, axis=1)
        return np.argmin(distances), np.min(distances)

    def update(self):
        sums_x = [0] * len(self.centroids)
        sums_y = [0] * len(self.centroids)
        counts = [0] * len(self.centroids)
        for datapoint in self.dataset:
            closest, _ = self.closest_centroid(datapoint)
            sums_x[closest] += datapoint[0]
            sums_y[closest] += datapoint[1]
            counts[closest] += 1

        converged = True
        for i in range(len(self.centroids)):
            new_position = [sums_x[i] / counts[i], sums_y[i] / counts[i]]
            if self.centroids[i][0] != new_position[0] and self.centroids[i][1] != new_position[1]:
                converged = False
            self.centroids[i] = new_position

        self.converged = converged

    def accuracy(self):
        counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for idx, datapoint in enumerate(self.dataset):
            closest, _ = self.closest_centroid(datapoint)
            counts[closest][math.floor(idx / 100)] += 1

        max_accuracy = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i != j and j != k and i != k:
                        acc = (counts[0][i] + counts[1][j] + counts[2][k]) / 300
                        if acc > max_accuracy:
                            max_accuracy = acc

        return max_accuracy

    def objective(self):
        obj = 0
        for idx, datapoint in enumerate(self.dataset):
            _, min_dist = self.closest_centroid(datapoint)
            obj += min_dist
        return obj
