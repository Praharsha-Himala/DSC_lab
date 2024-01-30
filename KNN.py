# importing necessary libraries
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# fetch dataset
abalone = fetch_ucirepo(id=1)

# data (as pandas dataframes)
X = np.array(abalone.data.features)
y = np.array(abalone.data.targets)
y = y.flatten()  # flattening for array compatibility


# print(X.shape, y.shape)
def distance(point1, point2):
    distance_between_points = np.linalg.norm(point1 - point2)
    return distance_between_points


class KNN:
    def __init__(self, k: int = 3):
        self.num_classes = None
        self.y_train = None
        self.X_train = None
        self.k = k

    # fitting returns number of classes in the data
    def fit(self, trainSet, labels):
        self.X_train, self.y_train = trainSet, labels
        self.num_classes = len(np.unique(labels))

    # prediction of each class based on k near neighbors
    def predict(self, testSet):
        labels = []
        for point in testSet:
            distances = [distance(point, x_train) for x_train in self.X_train]
            indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[indices]
            common_labels = np.bincount(k_nearest_labels).argmax()
            labels.append(common_labels)

        return np.array(labels)


# Splitting abalone dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=32)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


model = KNN(k=3)  # calling KNN model
model.fit(X_train, y_train)  # fitting

y_pred = model.predict(X_test)  # prediction

# Classification report
print(classification_report(y_pred, y_test))

# Confusion matrix to evaluate the model prediction class-wise
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("confusion matrix")
plt.xlabel("predicted label")
plt.ylabel("Actual label")
plt.show()

# Comparing the in-built library of KNN in sklearn
# from sklearn.neighbors import KNeighborsClassifier
#
# knn_sklearn = KNeighborsClassifier(n_neighbors=10)
#
# knn_sklearn.fit(X_train, y_train)
# predictions_sklearn = knn_sklearn.predict(X_test)
#
# print("Scikit-learn KNN classifier")
# print(classification_report(y_test, predictions_sklearn))
#
# cm_sklearn = confusion_matrix(y_test, predictions_sklearn)
#
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm_sklearn, annot=True, fmt="d")
# plt.title("confusion matrix")
# plt.xlabel("predicted label")
# plt.ylabel("Actual label")
# plt.show()
