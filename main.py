from matplotlib import pyplot
import numpy as np


class LinearRegression:
    def __init__(self, total_features):
        self.weights = np.zeros(total_features + 1)

    def predict(self, value):
        return self.weights * value

    def predict_features(self, features):
        return np.array(list(map(self.predict, features)))

    def get_error(self, features, target):
        total_features = len(features)
        predicted_values = self.predict_features(features)

        error = (predicted_values - target) ** 2

        # Divide by two to simplify derivative multiplication
        return error.sum() / (2 * total_features)

    # Gradient Descent, made by getting gradient of get_error function [f(x, y) = 1/N * sum(y - predict_values(x))]:
    def update_weights(self, features, target, rate):
        predicted_values = self.predict_features(features)

        error = target - predicted_values

        gradient_matrix = np.dot(-features.T, error) * rate
        gradient_matrix /= len(features)

        self.weights -= gradient_matrix

    def train(self, values, target, batches, rate=0.01):
        for i in range(batches):
            self.update_weights(values, target, rate)

            cost = self.get_error(values, target)

            if i % 10 == 0:
                print("iter={}\tweight={:.2f}\tbias={:.2f}\tcost={:.2f}".format(i, self.weights[:-1], self.weights[:-1], cost))


class DataManager:
    def __init__(self, features, target):
        self.target = target
        bias = np.ones(shape=(len(features), 1))
        self.features = np.append(features, bias, axis=1)

    def update_values(self, features, target):
        self.features = features
        self.target = target

    def normalize(self):
        for feature in self.features.T:
            feature_mean = np.mean(feature)
            feature_range = np.amax(feature) - np.amin(feature)
            feature -= feature_mean
            if feature_range:
                feature /= feature_range


sales = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
final = np.array([2.0, 3.0, 5.0, 9.0, 14.0])

regressor = LinearRegression(total_features=1)
data_manager = DataManager(sales, final)
data_manager.normalize()

print(data_manager.target)
print(regressor.predict_features(data_manager.features))

#pyplot.plot([x[0] for x in sales], final)
#pyplot.plot([x[0] for x in sales], regressor.predict_features(sales), 'r')
#pyplot.show()
