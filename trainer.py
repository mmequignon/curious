#!/home/pytorch/pytorch/sandbox/bin/python3

import random
import re

import torch
import numpy

from tic_tac_toe import TicTacToe


class Trainer():

    __slots__ = ["dataset", "trainset", "testset"]

    def __init__(self):
        """Gets each item from the dataset file and splits that dataset into
        two sets. One for training, one for testing.
        """
        split_ratio = 0.0001
        filename = "data/tic-tac-toe-dataset.txt"
        self.dataset = []
        with open(filename, "r") as f:
            for line in f:
                self.dataset.append(line.split())
        random.shuffle(self.dataset)
        split_indice = int(len(self.dataset) * split_ratio)
        self.trainset = self.dataset[:split_indice]
        self.testset = self.dataset[split_indice:]

    def get_leaves(self, parent):
        regex = re.compile("%s[0-9]+" % (parent))
        return [i for i in self.dataset if regex.match(i[0])]

    def best_branch(self, trunk, branches):
        """Depending on the parent given as argument, returns the best move.
        """
        # TODO: For the moment, this method do not evaluates losses
        # or nil options, in order to avoid a loss.
        childs = self.get_leaves(trunk)
        results = []
        for branch in branches:
            regex = re.compile("%s[0-9]*" % (trunk + str(branch)))
            leaves = [i for i in childs if regex.match(i[0])]
            # counters of victories for player one and two and for nil games
            one = two = nil = 0
            for leave in leaves:
                if leave[1].isnumeric():
                    if int(leave[1]) == 0:
                        one += 1
                    else:
                        two += 1
                else:
                    nil += 1
            # If the current player is One, we must maximze victories of One
            # else, Two.
            current = len(trunk) % 2 == 0 and one or two
            win = (current / len(leaves)) * 100
            results.append((win, branch))
        results.sort(reverse=True)
        return(int(results[0][-1]))

    def get_game_from_sequence(self, sequence):
        moves = [int(i) for i in sequence]
        game = TicTacToe()
        for move in moves:
            game.move(move)
            game.end_turn()
        return game

    def get_tensor_from_game(self, game):
        three_dim_array = [
            [[p[x + y] and 1 or 0 for y in range(3)] for
                x in range(0, 9, 3)] for p in game.table]
        n = numpy.array(three_dim_array)
        return torch.from_numpy(n)

    def split_table(self, table):
        return (table[:, :2, :2], table[:, :2, 1:],
                table[:, 1:, :2], table[:, 1:, 1:])


def chunker(data, size=2000):
    return [data[i:i+size] for i in range(0, len(data), size)]


if __name__ == "__main__":
    trainer = Trainer()
    chunk_size = 30
    # Traning phase
    chunks = chunker(trainer.dataset, chunk_size)
    for epoch, chunk in enumerate(chunks):
        current_loss = 0
        for sequence, winner in chunk:
            # Regarding the fact that each sequence of move provided by the
            # dataset represents an ended game, we must slice them.
            index = random.randrange(0, len(sequence) - 2)
            root = sequence[index]
            game = trainer.get_game_from_sequence(root)
            branches = game.valid_moves()
            best_branch = trainer.best_branch(root, branches)
            # TODO: for the moment, randomly selects branch.
            choice = random.choice(branches)
            if best_branch != choice:
                current_loss += 1
        rate = (current_loss / chunk_size) * 100
        # TODO: output in a file, for pyplot.
        print("iteration : %s, loss rate : %s" % (epoch, rate))
