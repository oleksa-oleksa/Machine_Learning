"""
Aufgabe 1. Implementierung eines k-NN-Classifiers
Implementieren Sie einen k-NN-Classifier in Python (inkl. numpy, pandas, matplotlib)
auf der Jupyter Notebook-Umgebung. Nutzen Sie den „ZIP code“-Datensatz mit den Trainings- daten als Referenz
für die Nachbarschaft. Testen Sie auf den Testdaten.

Geben Sie die Konfusionsmatrix und die Klassifikationsgenauigkeit aus.

Geben Sie mithilfe von Matplotlib ein paar der falsch klassifizierten Zahlen aus.
Testen Sie, bei welchem k ist der Klassifizierer am besten ist.
Was sind Vor- und Nachteile des k-NN-Classifiers?

===========================

About the dataset:
Normalized handwritten digits, automatically
scanned from envelopes by the U.S. Postal Service. The original
scanned digits are binary and of different sizes and orientations; the
images  here have been deslanted and size normalized, resulting
in 16 x 16 grayscale images (Le Cun et al., 1990).

The data are in two gzipped files, and each line consists of the digit
id (0-9) followed by the 256 grayscale values.

There are 7291 training observations and 2007 test observations,
distributed as follows:
         0    1   2   3   4   5   6   7   8   9 Total
Train 1194 1005 731 658 652 556 664 645 542 644 7291
 Test  359  264 198 166 200 160 170 147 166 177 2007

or as proportions:
         0    1   2    3    4    5    6    7    8    9
Train 0.16 0.14 0.1 0.09 0.09 0.08 0.09 0.09 0.07 0.09
 Test 0.18 0.13 0.1 0.08 0.10 0.08 0.08 0.07 0.08 0.09


Alternatively, the training data are available as separate files per
digit (and hence without the digit identifier in each row)

The test set is notoriously "difficult", and a 2.5% error rate is
excellent. These data were kindly made available by the neural network
group at AT&T research labs (thanks to Yann Le Cunn).

================

1. Open train and test datasets
2. Define X-axis: Features aka properties that we use for learning (independent variables)
3. Define Y-axis: Labels/Classes aka output our model should learn to produce (dependent variables)
4. Print/plot data for visual understanding



"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Classifier:
    """
    Accuracy refers to the closeness of a measured value to a standard or known value
    We want to measure the accuracy of our classifier.
    That is, we want to feed it a series of images whose contents are known and
    tally the number of times the model’s prediction matches the true content of an image.
    The accuracy is the fraction of images that the model classifies correctly.
    Because we are measuring our model’s accuracy, we have curated a set of images whose contents are known.
    That is, we have a true label for each image, which is encoded as a class-ID.

    Next, we can use NumPy’s vectorized logical operations, specifically ==, to get a boolean-valued array
    that stores True wherever the predicted labels match the true labels and False everywhere else.
    Recall that True behaves like 1 and False like 0. Thus, we can call np.mean on our resulting boolean-valued array
    to compute the number of correct predictions divided by the total number of predictions.
    """
    def accuracy(self, labels, predictions):
        # Compute the arithmetic mean along the specified axis
        return np.mean(labels == predictions)

    def confusion_matrix(self, labels, predictions):
        # set() method is used to convert any of the iterable to the distinct element
        # and sorted sequence of iterable elements, commonly called Set.
        size = len(set(labels))
        matrix = np.zeros((size, size))
        # map the similar index of multiple containers so that they can be used just using as single entity.
        for correct, predicted in zip(labels.astype(int), predictions):
            matrix[correct][predicted] += 1
        return matrix


class KNearestNeighbors(Classifier):
    # Use the Pythagorean theorem to determine the length of the hypotenuse
    def euclidean_distance(self, x_1, x_2):
        # Sum of array elements over a given axis.
        # print(np.sum((x_1 - x_2) ** 2, axis=1))
        return np.sum((x_1 - x_2) ** 2, axis=1)

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test, k):
        predictions = []
        for sample in X_test:
            distances = self.euclidean_distance(self.X, sample)
            # getting indexes of k first minimal elements
            indices = np.argpartition(distances, k)[:k]
            # taking labels by indexes
            votes = (self.y[indices]).astype(int)
            # the class with maximum votes => the class with minimal distance to k nearest neighbors
            winner = np.argmax(np.bincount(votes, minlength=10))
            predictions += [winner]
        print('Predictions for k=%d complete' % k)
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

X_train, y_train = training_data[:,1:-1], training_data[:,0]
X_test, y_test = test_data[:,1:], test_data[:,0]

print(training_data.shape)
print(test_data.shape)

# show_numbers(X_train)

model = KNearestNeighbors()
model.fit(X_train, y_train)

predictions_1 = model.predict(X_test, 1)
predictions_2 = model.predict(X_test, 2)
predictions_3 = model.predict(X_test, 3)

print(model.accuracy(y_test, predictions_1))
print(model.accuracy(y_test, predictions_2))
print(model.accuracy(y_test, predictions_3))

print(model.confusion_matrix(y_test, predictions_1))

misclassified = X_test[(predictions_1 != y_test)]
show_numbers(misclassified)

