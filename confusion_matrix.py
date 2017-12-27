#!/home/pytorch/pytorch/sandbox/bin/python3

import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]
confusion_mat = confusion_matrix(true_labels, pred_labels)

pyplot.imshow(confusion_mat, interpolation='nearest', cmap=pyplot.cm.gray)
pyplot.title("confusion matrix")
pyplot.colorbar()
ticks = numpy.arange(5)
pyplot.xticks(ticks, ticks)
pyplot.yticks(ticks, ticks)
pyplot.ylabel("True labels")
pyplot.xlabel("Predicted labels")
pyplot.savefig("plots/confusion_matrix.png")

targets = ["Class-0", "Class-1", "Class-2", "Class-3", "Class-4"]
print("\n", classification_report(
    true_labels, pred_labels, target_names=targets))
