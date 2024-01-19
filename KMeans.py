# import necessary libraries
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# loading iris data
data = load_iris()
iris = pd.DataFrame(data.data,
                    columns=['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm'])
# iris.head()
# loading two columns for performing KMeans
X = pd.DataFrame(data=[iris['sepal length in cm'], iris['sepal width in cm']])
X = X.T  # convert the table to rows and columns by transpose
X = X.to_numpy()  # converting dataframe to array


# Creating Class of KMeans algorithm
class KMeans:
    def __init__(self):
        pass

    # Fit for training data
    def fit(self, X, k: int, max_iters: int = 1000, tol: [int, float] = 0.5):

        # Assigning the data points to randomly picked points by taking min of distance_from_centeroid from each
        # centeroid
        def assign_to_clusters(X, centeroids, k: int):
            distance_from_centeroid = np.array([np.sqrt((centeroids[i] - X) ** 2) for i in range(k)])
            distance_from_centeroid = np.sum(distance_from_centeroid, axis=-1)
            labels = np.argmin(distance_from_centeroid, axis=0)
            return labels

        # Updating centeroids by taking mean of assigned cluster data points
        def update_centeroids(X, labels, k: int, centeroids):
            new_centeroids = np.array([np.mean(X[labels == i], axis=0) for i in range(k)])
            return new_centeroids

        # Calculating distance from centeroids
        def distance(data_points, centeroids):
            return np.mean(np.sqrt((data_points - centeroids) ** 2), axis=0)

        # fit algorithm begins here
        # Initialization by randomly picking 'k' centeroids
        centeroids = X[np.random.choice(range(len(X)), k, replace=False)]

        # Assigning clusters
        for iteration in range(max_iters):
            labels = assign_to_clusters(X, centeroids, k)
            # updating centeroids
            new_centeroids = update_centeroids(X, labels, k, centeroids)

            if distance(new_centeroids, centeroids).all() < tol:
                break
            centeroids = new_centeroids

        return labels, centeroids


model = KMeans()
y, final_centeroids = model.fit(X, 3, 1000, 0.01)
# After fitting relabel the y based on 'data.target' which can be used as true labels
target = data.target
# Plotting the cluster points and comparison
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# change axis based on given k value
axs[0].scatter(X[target == 0, 0], X[target == 0, 1], color='pink', label='Cluster 1')
axs[0].scatter(X[target == 1, 0], X[target == 1, 1], color='green', label='Cluster 2')
axs[0].scatter(X[target == 2, 0], X[target == 2, 1], color='yellow', label='Cluster 3')
axs[0].set_title("Actual")

axs[1].scatter(X[y == 0, 0], X[y == 0, 1], color='pink', label='Cluster 1')
axs[1].scatter(X[y == 1, 0], X[y == 1, 1], color='green', label='Cluster 2')
axs[1].scatter(X[y == 2, 0], X[y == 2, 1], color='yellow', label='Cluster 3')
axs[1].scatter(final_centeroids[:, 0], final_centeroids[:, 1], color='red', marker='X')
axs[1].set_title("predicted")

# printing classification report
# WARNING: Relabel the y before classification report
print(classification_report(y, target))

