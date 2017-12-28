#!/home/pytorch/pytorch/sandbox/bin/python3

import random
import re


class Trainer():

    def __init__(self):
        filename = "data/tic-tac-toe-dataset.txt"
        self.dataset = []
        with open(filename, "r") as f:
            for line in f:
                self.dataset.append(line.split())

    def get_childs(self, parent):
        regex = re.compile("[%s]{%s}.*" % (parent, len(parent)))
        return [i for i in self.dataset if regex.match(i[0])]


if __name__ == "__main__":
    trainer = Trainer()
    choice = random.choice(trainer.dataset)
    print(choice[0])
    parent = choice[0][:5]
    childs = trainer.get_childs(parent)
    print(len(childs))
