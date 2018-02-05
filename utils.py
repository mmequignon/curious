#!/home/pytorch/pytorch/sandbox/bin/python3

from matplotlib import pyplot
from contextlib import contextmanager


@contextmanager
def plot(data_types, filename=None, grid=False):
    data = {
        "plots": {},
    }
    for data_type in data_types:
        data["plots"][data_type[0][0]] = get_data_template(data_type)
    pyplot.show()
    for plot_name, vals in data["plots"].items():
        pyplot.subplot(vals["id"])
        if grid:
            pyplot.grid()
        pyplot.xlabel(vals["x_label"])
        pyplot.ylabel(vals["y_label"])
        vals["axes"] = pyplot.gca()
        vals["axes"].set_xlim(vals["x_limits"])
        vals["axes"].set_ylim(vals["y_limits"])
    p = Plot(pyplot, data)
    yield p
    if filename is not None:
        pyplot.savefig(filename)


class Plot():
    def __init__(self, plot, data):
        self.plot = plot
        self.data = data
        for plot_name, vals in self.data["plots"].items():
            vals["line"] = vals["axes"].plot(
                vals["x_data"],
                vals["y_data"],
                vals["color"])[0]

    def update(self, new_values):
        for plot_name, vals in new_values.items():
            for val_name, val in vals.items():
                self.data["plots"][plot_name][val_name].append(val)
        for plot_name in self.data["plots"]:
            self.data["plots"][plot_name]["line"].set_xdata(
                self.data["plots"][plot_name]["x_data"])
            self.data["plots"][plot_name]["line"].set_ydata(
                self.data["plots"][plot_name]["y_data"])
        self.plot.draw()
        self.plot.pause(1e-100)


def chunker(data, size=2000):
    return [data[i:i+size] for i in range(0, len(data), size)]


def get_data_template(data_type):
    template = {
        "id": data_type[3],
        "x_label": data_type[0][1],
        "y_label": data_type[0][0],
        "x_data": [],
        "y_data": [],
        "x_limits": data_type[1],
        "y_limits": data_type[2],
        "color": "g-"
    }
    if data_type[0][0] == "accuracy":
        template["color"] = "g-"
    if data_type[0][0] == "loss":
        template["color"] = "r-"
    return template
