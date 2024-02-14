import numpy as np
from scipy.stats import norm


class GaussianNaiveBayes:

    def __init__(self):
        self.X = None
        self.y = None
        self.classes = None
        self.parameters = None

    @staticmethod
    def gaussian_prob(x, mean, std):
        return norm.pdf(x, mean, std)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.parameters = []

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
        # calculate the mean and std
            parameters_c = [(np.mean(X_c[:, j], axis=0), np.std(X_c[:, j], axis=0)) for j in range(X_c.shape[1])]
            self.parameters.append(parameters_c)

    def predict(self, X):
        outputs = []
        for i, c in enumerate(self.classes):
            params_c = self.parameters[i]
            probs = np.array([self.gaussian_prob(X, mean, std) for mean, std in params_c])
            likelihood = np.prod(probs, axis=0)
            outputs.append(likelihood)

        predictions = self.classes[np.argmax(outputs, axis=0)]

        return predictions
