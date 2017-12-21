#!/home/pytorch/pytorch/sandbox/bin/python3

import numpy
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

from utilities.utilities import visualize_classifier

input_filename = "data/data_multivar_nb.txt"
data = numpy.loadtxt(input_filename, delimiter=',')
X, y = data[:, :-1], data[:, -1]
classifier = GaussianNB()
classifier.fit(X, y)
y_pred = classifier.predict(X)
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naïve Bayes classifier = ", round(accuracy, 2), "%")
visualize_classifier(classifier, X, y, "naive_bayes_1")

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)
accuracy = 100.0 * (y_test == y_test_pred).sum()
X_test.shape[0]
print("Accuracy of the new classifier =", round(accuracy, 2), "%")
visualize_classifier(classifier_new, X_test, y_test, "naive_bayes_2")

num_folds = 3
accuracy_values = cross_validation.cross_val_score(
    classifier, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy : " + str(round(100*accuracy_values.mean(), 2)) + "%")
precision_values = cross_validation.cross_val_score(
    classifier, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision : " + str(round(100 * precision_values.mean(), 2)) + "%")
recall_values = cross_validation.cross_val_score(
    classifier, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall : " + str(round(100 * recall_values.mean(), 2)) + "%")
f1_values = cross_validation.cross_val_score(
    classifier, X, y, scoring='f1_weighted', cv=num_folds)
print("F1 : " + str(round(100 * f1_values.mean(), 2)) + "%")
