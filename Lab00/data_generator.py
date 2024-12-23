import numpy as np
import matplotlib.pyplot as plt
import random

class DataGenerator:
    def __init__(self):
        self.x = []
        self.y = []
        self.n_clusters = 0

    def generate(self, num_clusters, points_per_cluster, std_dev):
        num = random.randint(1,40)
        rng = np.random.default_rng(num)
        clusters = []
        labels = []
        
        for i in range(num_clusters):
            mean = (i / num_clusters, i / num_clusters)
            cluster = rng.normal(loc=mean, scale=std_dev, size=(points_per_cluster, 2))
            clusters.append(cluster)
            labels.extend([i] * points_per_cluster)
        
        # Store in instance variables
        self.x = np.vstack(clusters)
        self.y = np.array(labels)
        self.n_clusters = num_clusters

    def plot_data(self):
        for i in range(self.n_clusters):
            plt.scatter(self.x[self.y == i, 0], self.x[self.y == i, 1], label=f'Cluster {i}', alpha=0.7)
        plt.title("Generated 2D Points")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.show()
    
    def rotate(self, angle):
        """Rotate the dataset in `self.x` by `angle` radians."""
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        self.x = self.x @ rotation_matrix.T

    def export_data(self, filename="dataset"):
        """Save `self.x` and `self.y` and self.n_clusters to files."""
        np.save(f"{filename}_x.npy", self.x)
        np.save(f"{filename}_y.npy", self.y)
        np.save(f"{filename}_clusters.npy", self.n_clusters)

    def import_data(self, filename="dataset"):
        """Load `self.x` and `self.y` and self.n_clusters from files."""
        self.x = np.load(f"{filename}_x.npy")
        self.y = np.load(f"{filename}_y.npy")
        self.n_clusters = np.load(f"{filename}_clusters.npy", self.n_clusters)
