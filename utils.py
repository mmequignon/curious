#!/home/pytorch/pytorch/sandbox/bin/python3

from plot import Plot
from matplotlib import pyplot
from contextlib import contextmanager


@contextmanager
def plot(data_types, filename=None, grid=False):
    data = {
        "plots": {},
    }
    for data_type in data_types:
        data["plots"][data_type[0][0]] = Plot.get_data_template(data_type)
    pyplot.show()
    for plot_name, vals in data["plots"].items():
        pyplot.subplot(vals["id"])
        if grid:
            pyplot.grid()
        pyplot.title(vals["title"])
        pyplot.xlabel(vals["x_label"])
        pyplot.ylabel(vals["y_label"])
        vals["axes"] = pyplot.gca()
        vals["axes"].set_xlim(vals["x_limits"])
        vals["axes"].set_ylim(vals["y_limits"])
    p = Plot(pyplot, data)
    yield p
    if filename is not None:
        pyplot.savefig(filename)


def chunker(data, size=2000):
    return [data[i:i+size] for i in range(0, len(data), size)]
