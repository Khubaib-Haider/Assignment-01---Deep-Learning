import numpy as np

class KNN(object):

    def __init__(self):
        pass

    def predict_label(self, dist, k=1):
        test = dist.shape[0]
        pred_y = np.zeros(test)
        for i in range(test):
            nearest_y = []
            nearest_y = self.y_train[np.argsort(dist[i])][0:k]
            pred_y[i] = np.bincount(nearest_y).argmax()
        return pred_y

    def training(self, X, y):

        self.X_train = X
        self.y_train = y

    def prediction(self, X, k):

        distance = self.compute_distances(X)

        return self.predict_labels(distance, k=k)

    def compute_distances(self, X):

        test = X.shape[0]
        train = self.X_train.shape[0]
        E_distance = np.zeros((test, train))

        E_distance = np.sqrt((X ** 2).sum(axis=1, keepdims=1) + (self.X_train ** 2).sum(axis=1) - 2 * X.dot(self.X_train.T))

        return E_distance

    def predict_output(self, distance, k=1):

        num_t = distance.shape[0]
        y_pred = np.zeros(num_t)
        for i in range(num_t):
            assign_y = []
            assign_y = self.y_train[np.argsort(distance[i])][0:k]
            assign_y = assign_y.astype(int)
            y_pred[i] = np.bincount(assign_y).argmax()
        return y_pred