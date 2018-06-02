#!/home/pytorch/pytorch/sandbox/bin/python3

import logging
import pickle
from plot import Plot
from matplotlib import pyplot
from contextlib import contextmanager

from nn_games.tic_tac_toe import Game, Net
from config import cuda_available


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


def save_net_parameters(net):
    with open("nets/%s-net" % Game.name, "wb") as f:
        pickle.dump(net, f)


def load_net_parameters():
    try:
        f = open("nets/%s-net" % Game.name, "rb")
        net = pickle.load(f)
    except OSError:
        net = Net()
    if cuda_available:
        net.cuda()
    return net


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


logger = get_logger()
