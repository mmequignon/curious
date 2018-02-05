#!/home/pytorch/pytorch/sandbox/bin/python3

import random

import torch
import numpy

from tic_tac_toe import TicTacToe
from utils import chunker, plot


class Trainer():

    __slots__ = ["dataset", "trainset", "testset"]

    def load_dataset(self, filename):
        ratios = {}
        with open(filename, "r") as f:
            for line in f:
                sequence, one, two, nil = line.split()
                ratios[sequence] = {
                    "one": int(one),
                    "two": int(two),
                    "nil": int(nil)
                }
        return ratios

    def __init__(self):
        """Gets each item from the dataset file and splits that dataset into
        two sets. One for training, one for testing.
        """
        filename = "data/tic-tac-toe-ratios-dataset.csv"
        self.dataset = self.load_dataset(filename)
        sequences = list(self.dataset.keys())
        split_ratio = 0.8
        split_indice = int(len(sequences) * split_ratio)
        self.trainset = sequences[:split_indice]
        self.testset = sequences[split_indice:]

    def evaluate(self, trunk, branches, _type="win"):
        results = []
        current = (
            _type == "nil" and "nil" or
            len(trunk) % 2 == 0 and "one" or
            "two")
        for branch in branches:
            data = self.dataset.get(trunk + str(branch), None)
            if data is None:
                results.append((0, branch))
                continue
            # counters of victories for player one and two and for nil games
            #  print(current, len(trunk), data)
            qty = sum(list(data.values()))
            win = (data[current] / qty) * 100
            results.append((win, int(branch)))
        results.sort(reverse=True)
        return(results[0])

    def best_branch(self, trunk, branches):
        """Depending on the parent given as argument, returns the best move.
        """
        ratio_win, branch = self.evaluate(trunk, branches)
        if ratio_win < 50:
            ratio_nil, nil = self.evaluate(trunk, branches, "nil")
            if ratio_nil > ratio_win:
                branch = nil
        return branch

    def train(self, trunk):
        game = trainer.get_game_from_sequence(trunk)
        branches = game.valid_moves()
        choice = random.choice(branches)
        branch = self.best_branch(trunk, branches)
        return choice != branch and 1 or 0

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


if __name__ == "__main__":
    trainer = Trainer()
    chunks = chunker(trainer.trainset)
    plots = [
        [("accuracy", "epoch"), (0, len(chunks)), (0, 100), 211],
        [("loss", "epoch"), (0, len(chunks)), (0, 100), 212],
    ]
    with plot(plots, filename="plots/accuracy.png", grid=True) as p:
        for epoch, chunk in enumerate(chunks):
            current_loss = 0
            args = []
            for i, sequence in enumerate(chunk):
                if len(sequence) < 3:
                    index = 0
                else:
                    index = random.randrange(0, len(sequence) - 2)
                root = "".join(sequence[:index])
                current_loss += trainer.train(root)
            loss = (current_loss / len(chunk)) * 100
            accuracy = 100 - loss
            new_values = {
                "accuracy": {"x_data": epoch, "y_data": accuracy},
                "loss": {"x_data": epoch, "y_data": loss},
            }
            p.update(new_values)
            print("epoch N°%s of size %s, loss = %s" % (
                epoch, len(chunk), loss))
    #  game = TicTacToe()
    #  init = random.randrange(0, 2)
    #  print(init)
    #  game.representation()
    #  while not game.game_is_over():
    #      moves = game.valid_moves()
    #      move = trainer.best_branch(game.get_sequence(), moves)
    #      game.move(move)
    #      game.end_turn()
    #      game.representation()
