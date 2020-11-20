"""
Implementation of Least-Squares Linear Regression
Using the closed-form expression from the lecture, implement Linear Regression in Python
(incl. Numpy, Pandas, Matplotlib) on a Jupyter Notebook.

Train on the training set of the "ZIP code"-Dataset and test on its test set.

(a) Print out the Confusion Matrix and the accuracy. (b) What is a good way of encoding the labels?
(c) What is the problem with using Linear Regression for Classification?
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Classifier:
    def accuracy(self, labels, predictions):
        # Compute the arithmetic mean along the specified axis
        return np.mean(labels == predictions)

    def confusion_matrix(self, labels, predictions):
        # set() method is used to convert any of the iterable to the distinct element
        # and sorted sequence of iterable elements, commonly called Set.
        size = len(set(labels))
        matrix = np.zeros((size, size))
        # predictions_list = [round(x) for x in predictions]

        # map the similar index of multiple containers so that they can be used just using as single entity.
        for correct, predicted in zip(labels.astype(int), np.array(predictions).astype(int)):
            matrix[correct][predicted] += 1
        return matrix


class LeastSquares(Classifier):
    def linear_distance(self, data, data_mean):
        return data - data_mean

    def mean_x(self):
        for test_sample in self.x:
            self.x_mean = np.mean(test_sample)

    def mean_y(self):
        for test_sample in self.y:
            self.y_mean = np.mean(test_sample)

    def fit(self, x, y):
        self.x = x
        self.y = y[:, np.newaxis]

    def predict(self, x_test):
        predictions = []
        x_mean = np.mean(self.x, axis=0, keepdims=True)
        # print("x_mean: ", x_mean)
        # print(self.x.shape, x_mean.shape, x_test.shape)
        y_mean = np.mean(self.y)
        print("y_mean: ", y_mean)


        delta_x = self.x - x_mean
        delta_y = self.y - y_mean

        delta_x_squared = delta_x ** 2
        distances_xy = delta_x * delta_y

        # y = intercept + slope * x
        slope = np.sum(distances_xy, axis=0) / np.sum(delta_x_squared, axis=0)

        intercept = y_mean - np.dot(slope, x_mean.T)
        print(slope.shape, x_mean.shape, intercept)
        for sample in x_test:
            y = slope * sample + intercept
            predictions.append(y)

        print('Linear regression complete')
        # print(predictions)
        return predictions


def show_numbers(X):
    num_samples = 90
    # Generates a random sample from a given 1-D array
    indices = np.random.choice(range(len(X)), num_samples)
    sample_digits = X[indices]

    fig = plt.figure(figsize=(20, 6))

    for i in range(num_samples):
        ax = plt.subplot(6, 15, i + 1)
        img = 255 - sample_digits[i].reshape((16, 16))
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    plt.show()


training_data = np.array(pd.read_csv('./datasets/zip.train', sep=' ', header=None, engine='python'))
test_data = np.array(pd.read_csv('./datasets//zip.test', sep=' ', header=None, engine='python'))

x_train, y_train = training_data[:,1:-1], training_data[:,0]
x_test, y_test = test_data[:,1:], test_data[:,0]

model = LeastSquares()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print(model.confusion_matrix(y_test, predictions))
misclassified = x_test[(predictions != y_test)]
show_numbers(misclassified)
