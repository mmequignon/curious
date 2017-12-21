#!/home/pytorch/pytorch/sandbox/bin/python3

import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot


def visualize_classifier(classifier, X, y, filename):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    mesh_step_size = 0.01
    x_vals, y_vals = numpy.meshgrid(
        numpy.arange(min_x, max_x, mesh_step_size),
        numpy.arange(min_y, max_y, mesh_step_size))
    output = classifier.predict(numpy.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    pyplot.figure()
    pyplot.pcolormesh(x_vals, y_vals, output, cmap=pyplot.cm.gray)
    pyplot.scatter(
        X[:, 0], X[:, 1],
        c=y, s=75, edgecolors="black",
        linewidth=1, cmap=pyplot.cm.Paired)
    pyplot.xlim(x_vals.min(), x_vals.max())
    pyplot.ylim(y_vals.min(), y_vals.max())
    pyplot.xticks((
        numpy.arange(int(X[:, 0].min() - 1),
                     int(X[:, 0].max() + 1),
                     1.0)))
    pyplot.yticks((
        numpy.arange(int(X[:, 1].min() - 1),
                     int(X[:, 1].max() + 1),
                     1.0)))
    pyplot.savefig("plots/%s.png" % filename)
